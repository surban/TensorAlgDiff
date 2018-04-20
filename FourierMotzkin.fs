/// Foruier-Motzkin elimination method for solving system of inequalities.
/// This algorithm is only practical for very small systems.
module FourierMotzkin

open Tensor
open Tensor.Algorithm


/// A range for a particular element of x in a system of inequalities.
/// The low limits and high limits for x[Idx] are given as follows:
/// Low limits:  x[Idx] >= BLow  .* b - SLow  .* x.[Idx+1L..]
/// High limits: x[Idx] <= BHigh .* b - SHigh .* x.[Idx+1L..]
/// If a low limits is larger that a high limit, then no solution exists.
type Range = {
    /// Index of x element this solution is for.
    Idx:        int64
    /// Matrix to multiply b with to obtain low limits.
    BLow:       Tensor<Rat>
    /// Matrix to multiply b with to obtain high limits.
    BHigh:      Tensor<Rat>
    /// Matrix to multiply x.[Idx+1L..] with to obtain low limits.
    SLow:       Tensor<Rat>
    /// Matrix to multiply x.[Idx+1L..] with to obtain high limits.
    SHigh:      Tensor<Rat>
}

/// Solution element.
type Solution =
    /// Specifies range for a particular element of x.
    | Range of Range
    /// Specifies that system only has solution if B .* b <= 0.
    | Feasibility of B:Tensor<Rat>

/// A possible solution to a system of inequalities.
type Solutions = Solution list

/// Solves a system of inequalities of the form A .* x >= b for arbitrary b.
let solve (A: Tensor<Rat>) : Solutions =   
    let m, n = 
        match A.Shape with
        | [m; n] -> m, n
        | _ -> invalidArg "A" "A must be a matrix"
    let needFeasibilityCheck = 
        A ==== Rat.Zero |> Tensor.allAxis 1 |> Tensor.any 

    /// Elimination step.
    let rec eliminate (rA: Tensor<Rat>) (rB: Tensor<Rat>) rAs rBs =
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
                eliminate rA rB (rA::rAs) (rB::rBs) 
            elif Tensor.all (rA.[*, k] ==== Rat.One) || Tensor.all (rA.[*, k] ==== Rat.MinusOne) then
                // the coefficients of x_k are all +1 or -1
                //printfn "all x_k=+1 or all x_k=-1"
                eliminate (rA.M(zRows, NoMask)) (rB.M(zRows, NoMask)) (rA::rAs) (rB::rBs)
            elif Tensor.all ((rA.[*,k] ==== Rat.Zero) |||| (rA.[*,k] ==== Rat.One)) ||
                 Tensor.all ((rA.[*,k] ==== Rat.Zero) |||| (rA.[*,k] ==== Rat.MinusOne)) then
                // the coefficients of x_k are a mix of 0 and +1 or a mix of 0 and -1
                //printfn "x_k is mix of 0 and +1 or mix of 0 and -1"
                eliminate (rA.M(zRows, NoMask)) (rB.M(zRows, NoMask)) (rA::rAs) (rB::rBs) 
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
                eliminate nextRA nextRB (rA::rAs) (rB::rBs) 
        else
            let feasibility = 
                if needFeasibilityCheck && rB.Shape.[0] > 0L then [Feasibility rB]
                else []
            backSubst rAs rBs feasibility

    /// Backsubstitution step.
    and backSubst rAs rBs sols =
        match rAs, rBs with
        | rA::rAs, rB::rBs ->
            let k = List.length rAs |> int64

            // split B for lower and upper limit of x_j
            let Blow = rB.M(rA.[*,k] ==== Rat.One, NoMask)
            let Bhigh = -rB.M(rA.[*,k] ==== Rat.MinusOne, NoMask)

            // substitute the values into the system: x = [0; ...; 0; v_j; ...; v_n]
            // solution: B .* y - A .* x.[j..n]
            let S = -rA.[*, k+1L..]
            //printfn "S for %d=\n%A" k S

            // split C for lower and upper limit of x_j
            let Slow = S.M(rA.[*,k] ==== Rat.One, NoMask)
            let Shigh = -S.M(rA.[*,k] ==== Rat.MinusOne, NoMask)          

            let sol = Range {
                Idx=k
                BLow=Blow; BHigh=Bhigh
                SLow=Slow; SHigh=Shigh
            } 
            backSubst rAs rBs (sol::sols)
        | _ -> sols

    eliminate A (HostTensor.identity m) [] [] |> List.rev


/// Checks system for feasibility.
let feasible (fs: Tensor<Rat>) (b: Tensor<Rat>) =
    match b.Shape with
    | [l] when l = fs.Shape.[1] -> ()
    | _ -> invalidArg "b" "b has wrong size"
    Rat.Zero >>== fs .* b |> Tensor.all
    

/// Returns the range (xMin, xMax) for x.[sol.Idx] so that xMin <= x.[sol.Idx] <= xMax given x.[sol.Idx+1L..].
/// If xMin > xMax, then no solution exists.
let range (sol: Range) (b: Tensor<Rat>) (xRight: Tensor<Rat>) =
    match b.Shape with
    | [l] when l = sol.BLow.Shape.[1] -> ()
    | _ -> invalidArg "b" "b has wrong size"
    match xRight.Shape with
    | [l] when l = sol.SLow.Shape.[1] -> ()
    | _ -> invalidArg "xRight" "wrong number of substitution variables"    

    let lows = sol.BLow .* b + sol.SLow .* xRight
    let highs = sol.BHigh .* b + sol.SHigh .* xRight
    Tensor.max lows, Tensor.min highs

