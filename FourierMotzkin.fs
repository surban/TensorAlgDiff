/// Foruier-Motzkin elimination method for solving system of inequalities.
/// This algorithm is only practical for very small systems.
module FourierMotzkin

open Tensor


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
/// After the solution is computed, use the functions 'range' and 'subst' to obtain the allowed
/// ranges of the variables and substitute values for them in the inequality system.
/// Returns None if the inequality system is infeasible.
let solve (A: Tensor<Rat>) (b: Tensor<Rat>) =   
    let m, n = A.Shape.[0], A.Shape.[1]
    let rec eliminate (rA: Tensor<Rat>) (rb: Tensor<Rat>) rAs rbs arb =
        let k = List.length rAs |> int64
        //printfn "Elimination step %d:" k
        //printfn "rA=\n%A" rA
        //printfn "rb=\n%A" rb
        //printfn "arbitrary: %A" arb
        if k < n then
            let rA, rb = Tensor.copy rA, Tensor.copy rb
            let zRows = rA.[*, k] ==== Rat.Zero
            let nzRows = ~~~~zRows

            // Divide rows so that x_k = +1 or x_k = -1 or x_k = 0 in each inequality.     
            let facs = abs (rA.M(nzRows, NoMask).[*, k]) 
            rA.M(nzRows, NoMask) <- rA.M(nzRows, NoMask) / facs.[*, NewAxis]
            rb.M(nzRows) <- rb.M(nzRows) / facs

            //printfn "after division:"
            //printfn "rA=\n%A" rA
            //printfn "rb=\n%A" rb

            // Check condition of x_k.
            if Tensor.all (rA.[*, k] ==== Rat.Zero) then
                // all the coefficients of x_k are zero, thus it is arbitrary
                //printfn "all x_k=0"
                eliminate rA rb (rA::rAs) (rb::rbs) (arb |> Set.add k)
            elif Tensor.all (rA.[*, k] ==== Rat.One) || Tensor.all (rA.[*, k] ==== Rat.MinusOne) then
                // the coefficients of x_k are all +1 or -1
                //printfn "all x_k=+1 or all x_k=-1"
                eliminate (rA.M(zRows, NoMask)) (rb.M(zRows)) 
                          (rA::rAs) (rb::rbs) (arb |> Set.union (Set [k+1L .. n-1L]))
            elif Tensor.all ((rA.[*,k] ==== Rat.Zero) |||| (rA.[*,k] ==== Rat.One)) ||
                 Tensor.all ((rA.[*,k] ==== Rat.Zero) |||| (rA.[*,k] ==== Rat.MinusOne)) then
                // the coefficients of x_k are a mix of 0 and +1 or a mix of 0 and -1
                //printfn "x_k is mix of 0 and +1 or mix of 0 and -1"
                eliminate (rA.M(zRows, NoMask)) (rb.M(zRows)) (rA::rAs) (rb::rbs) arb
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
                let nextRb = 
                    List.allPairs pRows nRows            
                    |> List.map (fun (p, n) -> rb.[p..p] + rb.[n..n])
                    |> List.append [rb.M(zRows)]
                    |> Tensor.concat 0      
                eliminate nextRA nextRb (rA::rAs) (rb::rbs) arb
        else
            // Feasibility check
            if Tensor.any (rb >>>> Rat.Zero) then None
            else 
                Some {
                    ReducedA = rAs
                    ReducedB = rbs
                    Arbitrary = arb
                    Values = []
                }

    eliminate A b [] [] Set.empty


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

