/// Foruier-Motzkin elimination method for solving system of inequalities.
/// This algorithm is only practical for very small systems.
module FourierMotzkin

open Tensor


/// A possible solution to a system of inequalities without knowing the right-hand sides.
type Presolution = {
    /// right hand side matrix of fully reduced system
    FeasibilityB:   Tensor<Rat> 
    /// list of left sides of reduced systems
    ReducedA:   Tensor<Rat> list
    /// list of right side matrices of reduced systems
    ReducedB:   Tensor<Rat> list
    /// set of variable indicies that are arbitrary
    Arbitrary:  Set<int64>
}


/// A solution to a system of inequalities.
/// Use 'range' to obtain valid ranges and 'subst' to substitute a value for a variable.
type Solution = {
    /// list of left sides of reduced systems
    ReducedA:   Tensor<Rat> list
    /// list of right sides of reduced system
    ReducedB:   Tensor<Rat> list
    /// set of variable indicies that are arbitrary
    Arbitrary:  Set<int64>
    /// variable values for backsubstitution
    Values:     Rat list
}


/// Solves a system of inequalities of the form A .* x >= b.
/// A must not contain rows of zeros.
/// b is specifed using the function solve.
let presolve (A: Tensor<Rat>) =   
    let m, n = 
        match A.Shape with
        | [m; n] -> m, n
        | _ -> invalidArg "A" "A must be a matrix"
    if A ==== Rat.Zero |> Tensor.allAxis 1 |> Tensor.any then
        invalidArg "A" "A must not contain rows of zeros"
        
    let rec eliminate (rA: Tensor<Rat>) (rB: Tensor<Rat>) rAs rBs arb =
        let k = List.length rAs |> int64
        //printfn "Elimination step %d:" k
        //printfn "rA=\n%A" rA
        //printfn "rb=\n%A" rb
        //printfn "arbitrary: %A" arb
        if k < n then
            let rA, rB = Tensor.copy rA, Tensor.copy rB
            let zRows = rA.[*, k] ==== Rat.Zero
            let nzRows = ~~~~zRows

            // Divide rows so that x_k = +1 or x_k = -1 or x_k = 0 in each inequality.     
            let facs = abs (rA.M(nzRows, NoMask).[*, k..k]) 
            rA.M(nzRows, NoMask) <- rA.M(nzRows, NoMask) / facs
            rB.M(nzRows, NoMask) <- rB.M(nzRows, NoMask) / facs

            //printfn "after division:"
            //printfn "rA=\n%A" rA
            //printfn "rb=\n%A" rb

            // Check condition of x_k.
            if Tensor.all (rA.[*, k] ==== Rat.Zero) then
                // all the coefficients of x_k are zero, thus it is arbitrary
                //printfn "all x_k=0"
                eliminate rA rB (rA::rAs) (rB::rBs) (arb |> Set.add k)
            elif Tensor.all (rA.[*, k] ==== Rat.One) || Tensor.all (rA.[*, k] ==== Rat.MinusOne) then
                // the coefficients of x_k are all +1 or -1
                //printfn "all x_k=+1 or all x_k=-1"
                eliminate (rA.M(zRows, NoMask)) (rB.M(zRows, NoMask)) 
                          (rA::rAs) (rB::rBs) (arb |> Set.union (Set [k+1L .. n-1L]))
            elif Tensor.all ((rA.[*,k] ==== Rat.Zero) |||| (rA.[*,k] ==== Rat.One)) ||
                 Tensor.all ((rA.[*,k] ==== Rat.Zero) |||| (rA.[*,k] ==== Rat.MinusOne)) then
                // the coefficients of x_k are a mix of 0 and +1 or a mix of 0 and -1
                //printfn "x_k is mix of 0 and +1 or mix of 0 and -1"
                eliminate (rA.M(zRows, NoMask)) (rB.M(zRows, NoMask)) (rA::rAs) (rB::rBs) arb
            else
                //printfn "x_k has +1 and -1"
                // there is at least one pair of inequalities with a +1 and a -1 coefficient for x_k
                let pRows = rA.[*, k] ==== Rat.One |> Tensor.trueIdx |> Tensor.flatten |> HostTensor.toList
                let nRows = rA.[*, k] ==== Rat.MinusOne |> Tensor.trueIdx |> Tensor.flatten |> HostTensor.toList
                // for each pair augment the reduced system by their sum
                let nextRA = 
                    List.allPairs pRows nRows            
                    |> List.map (fun (p, n) -> rA.[p..p, *] + rA.[n..n, *])
                    |> List.append [rA.M(zRows, NoMask)]
                    |> Tensor.concat 0
                let nextRB = 
                    List.allPairs pRows nRows            
                    |> List.map (fun (p, n) -> rB.[p..p, *] + rB.[n..n, *])
                    |> List.append [rB.M(zRows, NoMask)]
                    |> Tensor.concat 0      
                eliminate nextRA nextRB (rA::rAs) (rB::rBs) arb
        else
            {
                FeasibilityB = rB
                ReducedA = rAs
                ReducedB = rBs
                Arbitrary = arb
            }

    let B = HostTensor.identity m
    eliminate A B [] [] Set.empty



/// Low limits:  x[Idx] >= BiasTransformLow  .* b - SubstLow  .* x.[Idx+1..]
/// High limits: x[Idx] <= BiasTransformHigh .* b - SubstHigh .* x.[Idx+1..]
type GenSolution = {
    Idx:                int64
    BiasLow:            Tensor<Rat>
    BiasHigh:           Tensor<Rat>
    SubstLow:           Tensor<Rat>
    SubstHigh:          Tensor<Rat>
}


let genSolve (ps: Presolution) =

    List.zip ps.ReducedA ps.ReducedB
    |> List.mapi (fun i (rA, rB) ->
        // active x_j
        let j = ps.ReducedA.Length - i - 1 |> int64    

        // split B for lower and upper limit of x_j
        let Blow = rB.M(rA.[*,j] ==== Rat.One, NoMask)
        let Bhigh = -rB.M(rA.[*,j] ==== Rat.MinusOne, NoMask)

        // substitute the values into the system: x = [0; ...; 0; v_j; ...; v_n]
        // solution: B .* y - A .* x.[j..n]
        let C = -rA.[*, j+1L..]
        printfn "C for %d=\n%A" j C

        // split C for lower and upper limit of x_j
        let Clow = C.M(rA.[*,j] ==== Rat.One, NoMask)
        let Chigh = -C.M(rA.[*,j] ==== Rat.MinusOne, NoMask)
       
        {
            Idx=j
            BiasLow=Blow
            BiasHigh=Bhigh
            SubstLow=Clow
            SubstHigh=Chigh
        }
    )


let genSubst (gs: GenSolution) (b: Tensor<Rat>) (x: Tensor<Rat>) =
    match b.Shape with
    | [l] when l = gs.BiasLow.Shape.[1] -> ()
    | _ -> invalidArg "b" "wrong number of bias variables"
    match x.Shape with
    | [l] when l = gs.SubstLow.Shape.[1] -> ()
    | _ -> invalidArg "x" "wrong number of substitution variables"    

    let lows = gs.BiasLow .* b + gs.SubstLow .* x
    let highs = gs.BiasHigh .* b + gs.SubstHigh .* x
    Tensor.max lows, Tensor.min highs


/// Specifies b in a presolved system.
/// Returns a solution if system is feasible and None otherwise.
/// Use the functions 'range' and 'subst' to obtain the allowed
/// ranges of the variables and substitute values for them in the inequality system.
let solve (ps: Presolution) (b: Tensor<Rat>) =
    let rB = ps.FeasibilityB .* b
    if Tensor.any (rB >>>> Rat.Zero) then None
    else 
        Some {
            ReducedA = ps.ReducedA
            ReducedB = ps.ReducedB |> List.map (fun B -> B .* b)
            Arbitrary = ps.Arbitrary
            Values = []
        }
    

/// Gets the index j of the active variable x_j.
/// Returns -1L if no variable is active, i.e. all variables have been substituted.
/// A solution returned by 'solve' starts with the last element of x being active.
/// Each call to 'subst' moves forward by one element.
let active (sol: Solution) =
    int64 sol.ReducedA.Length - 1L


/// Gets the allowed range of active variable.
/// Returns a tuple (low, high).
let range (sol: Solution) =
    let j = active sol
    if j < 0L then failwith "no variable active"
    if sol.Arbitrary |> Set.contains j then
        Rat.NegInf, Rat.PosInf
    else
        //printfn "Calculating range for x_%d:" j
        // substitute the values into the system: x = [0; ...; 0; v_j; ...; v_n]
        let A, b = List.head sol.ReducedA, List.head sol.ReducedB
        //printfn "A=\n%A" A
        //printfn "b=\n%A" b
        let x = Tensor.concat 0 [HostTensor.zeros [j+1L]; HostTensor.ofList sol.Values] 
        //printfn "x=\n%A" x
        let s = b - A .* x
        //printfn "s=\n%A" s
        // if coefficient of x_j is +1, then line of s is lower limit for x_j
        // if coefficient of x_j is -1, then line of -s is upper limit for x_j
        let low = s.M(A.[*,j] ==== Rat.One) |> Tensor.max
        let high = -s.M(A.[*,j] ==== Rat.MinusOne) |> Tensor.min
        low, high


/// Substitutes the active variable x_j with the given value, which must be within its allowed range,
/// and makes the variables x_{j-1} active.
let subst (value: Rat) (sol: Solution) =
    let low, high = range sol
    if not (low <= value && value <= high) then
        failwithf "value %A out of required range: %A <= x_%d <= %A" value low (active sol) high
    {sol with
        ReducedA = List.tail sol.ReducedA
        ReducedB = List.tail sol.ReducedB
        Values = value :: sol.Values}

