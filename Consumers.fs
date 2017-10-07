module Consumers

open System.Numerics
open Tensor
open Tensor.Algorithms


/// Modulo operator returning always non-negative values.
let inline (%%) (x: ^T) (d: ^T) : ^T =
    let r = x % d
    if r < LanguagePrimitives.GenericZero then d + r
    else r

let gcd a b =
    BigInteger.GreatestCommonDivisor (a, b)

let lcm a b =
    (a * b) / gcd a b |> abs


/// Tuple of (low, high) range.
type Range = int64 * int64


/// Necessary information to compute all xs consuming an y.
/// The xs are specified in terms of a base point dependant on y and a nullspace.
/// We have: x = YToX .* y + Nullspace .* z where z is an integer vector.
type ConsumerInfo = {
    /// Solvability .* Y = 0 for system to be solvable
    Solvability:        Tensor<bigint>
    /// Matrix mapping y to a particular solution x.
    YToX:               Tensor<Rat>
    /// Matrix of null-space basis.
    Nullspace:          Tensor<bigint> 
    /// Presolution of left side of constaint system.
    /// Right side is given by: [low - YToX .* y; -high + YToX .* y]
    ConstraintsLeft:    FourierMotzkin.Presolution
    /// X ranges.
    Ranges:             Range list
}



/// Computes information to determine all integer x in the equation y = M .* x
/// where x is in the specified ranges.
let compute (m: Tensor<bigint>) (rngs: Range list) =
    let ny, nx = m.Shape.[0], m.Shape.[1]
    if List.length rngs <> int nx then failwith "incorrect range count"
    let xLow = rngs |> List.map fst |> HostTensor.ofList
    let xHigh = rngs |> List.map snd |> HostTensor.ofList

    /// Invert matrix M over integers, giving inverse I, solvability S and nullspace N.
    let I, S, N = LinAlg.integerInverse m
    printfn "Inversion of M=\n%A\n gives inverse I=\n%A\n and nullspace N=\n%A" m I N

    // Constraints are:
    // x_i = YToX_i* .* y + Nullspace_i* .* z >= low_i
    // x_i = YToX_i* .* y + Nullspace_i* .* z <= high_i
    // Thus: Nullspace_i* .* z >=  low_i  - YToX_i* .* y
    //      -Nullspace_i* .* z >= -high_i + YToX_i* .* y
    let cLeft = Tensor.concat 0 [N; -N] 
    let cPresol = FourierMotzkin.presolve (Tensor.convert<Rat> cLeft)

    {
        Solvability     = S
        YToX            = I
        Nullspace       = N
        ConstraintsLeft = cPresol
        Ranges          = rngs
    }    


/// Get all x consuming specified y.
let get (ci: ConsumerInfo) (y: Tensor<bigint>) =
    let toRat = Tensor.convert<Rat>
    let toInt = Tensor.convert<int>

    // Base solution.
    let y = toRat y
    let xBase = ci.YToX .* y |> Tensor.convert<bigint>

    // Build biases of constraint system.
    let lows, highs = List.unzip ci.Ranges
    let lows, highs = HostTensor.ofList lows, HostTensor.ofList highs
    let cRight1 =  toRat lows  - ci.YToX .* y
    let cRight2 = -toRat highs + ci.YToX .* y
    let cRight = Tensor.concat 0 [cRight1; cRight2]
    //printfn "cRight=\n%A" cRight

    match FourierMotzkin.solve ci.ConstraintsLeft cRight with
    | Some sol ->
        let rec doIter sol z = seq {
            let j = FourierMotzkin.active sol
            if j >= 0L then
                let zjLow, zjHigh = FourierMotzkin.range sol
                let zjLow = zjLow |> ceil |> int
                let zjHigh = zjHigh |> floor |> int                
                for zj in zjLow .. zjHigh do
                    yield! (doIter (sol |> FourierMotzkin.subst (Rat zj)) (zj :: z))
            else
                let z = z |> HostTensor.ofList |> Tensor.convert<bigint>
                let x = xBase + ci.Nullspace .* z |> Tensor.convert<int64>
                yield (HostTensor.toList x)
        }
        doIter sol []
    | None -> Seq.empty



