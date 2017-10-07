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


/// Faith of an x dimension.
type DimFaith =
    /// Dimension is free with given index
    | Free of int
    /// Dimension is dependant with given index
    | Dependant of int


/// Necessary information to compute all xs consuming an y.
/// xFree are free integer coordinates limited by the constraints and divisibility constraints.
/// xDep are dependant coordiantes given by: xDep = XFreeToXDep .* XFree + YToXDep .* Y.
/// Constraints 1 are of the form: XFreeToC .* xFree + YToC .* Y >= cb.
/// Constraints 2 are of the form: XFreeToC .* xFree + YToC .* Y <= cb.
/// Divisibility constraints are of the form: XFree = YToRem .* Y (mod RDiv).
type ConsumerInfo = {
    /// Solvability .* Y = 0 for system to be solvable
    Solvability:       Tensor<bigint>
    /// What has become of a coordinate?
    XDimFaith:         DimFaith list
    /// Matrix mapping from free to dependant coordinates
    XFreeToXDep:       Tensor<Rat>
    /// Matrix mapping from input coordinates to dependant coordinates
    YToXDep:           Tensor<Rat>
    /// Matrix mapping from free coordinates to constraints
    XFreeToC1:         Tensor<Rat> 
    /// Matrix mapping from input coordinates to constraints
    YToC1:             Tensor<Rat>
    /// Constraint bias 
    CB1:               Tensor<int64> 
    /// Matrix mapping from free coordinates to constraints
    XFreeToC2:         Tensor<Rat> 
    /// Matrix mapping from input coordinates to constraints
    YToC2:             Tensor<Rat>
    /// Constraint bias 
    CB2:               Tensor<int64>     
    /// Presolution of constaint system
    CPresolution:      FourierMotzkin.Presolution
    /// Matrix mapping from input coordinates to division remainders
    YToRem:            Tensor<Rat>
    /// Dividers
    RDiv:              Tensor<bigint>       
}


/// Tuple of (low, high) range.
type Range = int64 * int64


/// Computes information to determine all integer x in the equation y = M .* x
/// where x is in the specified ranges.
let compute (m: Tensor<bigint>) (rngs: Range list) =
    let ny, nx = m.Shape.[0], m.Shape.[1]
    if List.length rngs <> int nx then failwith "incorrect range count"
    let xLow = rngs |> List.map fst |> HostTensor.ofList
    let xHigh = rngs |> List.map snd |> HostTensor.ofList

    /// Invert matrix M over integers, giving inverse I, solvability S and nullspace N.
    let I, S, N = LinAlg.integerInverse m

    // We need as many input dimensions as there are columns, the remaining dimensions will become dependants.
    // By removing rows with minimum absolute value first, it is ensured that zero rows are always removed.
    let deps = 
        N 
        |> abs 
        |> Tensor.convert<Rat>
        |> Tensor.minAxis 1
        |> HostTensor.toList
        |> List.indexed
        |> List.sortBy snd
        |> List.map fst
        |> List.take (N.Shape.[0] - N.Shape.[1] |> int)
        |> List.sort
    let free =
        [0 .. int N.Shape.[0]-1]
        |> List.filter (fun r -> not (List.contains r deps))

    // Faith of dimensions.
    let dimFaith =
        [0 .. int nx-1]
        |> List.map (fun d ->
            match free |> List.tryFindIndex ((=) d), deps |> List.tryFindIndex ((=) d) with
            | Some fd, None -> Free fd
            | None, Some dd -> Dependant dd
            | _ -> failwith "dimension cannot be both free and dependant")

    // Split inverse and nullspace into dependant and free partitions.
    let getRows rowList (V: Tensor<_>) =
        rowList
        |> List.map (fun r -> V.[int64 r .. int64 r, *])
        |> Tensor.concat 0   
    let ID = getRows deps I
    let IF = getRows free I
    let ND = getRows deps N
    let NF = getRows free N
    //printMat "Dependants ND" ND
    //printMat "Free       NF" NF

    // Write the dependants as a function of the free.
    // This is done by solving the free for the basis factors and then inserting into the dependants.
    let NFI, _, _ = LinAlg.integerInverse NF
    let NDF = Tensor.convert<Rat> ND .* NFI
    //printMat "Dependants given Free NDF" NDF

    // Compute LCM of inverse of free nullspace, since all free coordinates must be divisable by it.
    let FD = NFI |> Tensor.foldAxis (fun l v -> lcm l v.Dnm) (HostTensor.scalar bigint.One) 0
    //printMat "Free divisibiltiy requirements FD" FD

    // Compute range constraints in terms of free coordinates.
    let XFreeId = HostTensor.identity (int64 free.Length)
    let XFreeZeros = HostTensor.zeros [int64 free.Length; ny]
    let XFreeConstr = Tensor.concat 0 [XFreeId; NDF]
    let YConstr = Tensor.concat 0 [XFreeZeros; ID - NDF .* IF]

    // Build and presolve constraint system.
    let C = Tensor.concat 0 [XFreeConstr; -XFreeConstr]
    let Cpresol = FourierMotzkin.presolve C

    {
        Solvability     = S
        XDimFaith       = dimFaith
        XFreeToXDep     = NDF
        YToXDep         = ID - NDF .* IF 
        XFreeToC1       = XFreeConstr
        YToC1           = YConstr
        CB1             = xLow
        XFreeToC2       = XFreeConstr
        YToC2           = YConstr
        CB2             = xHigh
        CPresolution    = Cpresol
        YToRem          = IF
        RDiv            = FD
    }
    



/// Get all x consuming specified y.
let get (ci: ConsumerInfo) (y: Tensor<bigint>) =
    let toRat = Tensor.convert<Rat>
    let toInt = Tensor.convert<int>

    // Compute remainder constraints.
    let freeDiv = ci.RDiv |> toInt
    let freeRem = ci.YToRem .* toRat y |> toInt

    // Build biases of constraint system.
    let b1 = toRat ci.CB1 - ci.YToC1 .* toRat y
    let b2 = -toRat ci.CB2 + ci.YToC2 .* toRat y
    let b = Tensor.concat 0 [b1; b2]
    //printMat "b" b

    match FourierMotzkin.solve ci.CPresolution b with
    | Some sol ->
        let rec doSum sol xFree = seq {
            let j = FourierMotzkin.active sol
            if j >= 0L then
                let low, high = FourierMotzkin.range sol
                let low = low |> ceil |> int
                let high = high |> floor |> int
                
                let incr, rem = freeDiv.[[j]], freeRem.[[j]]
                let rem = rem %% incr
                //printfn "x_%d:  increment=%d   remainder=%d" j incr rem

                //printfn "low before adjustment: %d" low
                let lowDiff = rem - (low %% incr)
                let lowDiff = if lowDiff < 0 then lowDiff + incr else lowDiff
                let low = low + lowDiff
                //printfn "low after adjustment: %d" low

                //printfn "high before adjustment: %d" high
                let highDiff = rem - (high %% incr)
                let highDiff = if highDiff > 0 then highDiff - incr else highDiff
                let high = high + highDiff
                //printfn "high after adjustment: %d" high

                for i in low .. incr .. high do
                    yield! (doSum (sol |> FourierMotzkin.subst (Rat i)) (i :: xFree))
            else
                let xFree = xFree |> HostTensor.ofList |> Tensor.convert<int64>
                let xDep = ci.XFreeToXDep .* toRat xFree + ci.YToXDep .* toRat y |> Tensor.convert<int64>

                // reassemble
                let x =
                    ci.XDimFaith
                    |> List.map (function | Free i -> xFree.[[int64 i]]
                                          | Dependant i -> xDep.[[int64 i]])                    
                yield x
        }
        doSum sol []
    | None -> Seq.empty



