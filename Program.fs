
open System
open System.Numerics
open PLplot
open Tensor
open Tensor.Algorithms




let printMat title mat =
    let prefix = title + " = "
    let mp = sprintf "%A" mat
    let mp = mp.Replace("\n", "\n" + String.replicate prefix.Length " ")
    printfn "%s%s" prefix mp


let rectangle (pl: PLStream) (x1, y1) (x2, y2) =
    pl.line ([|x1; x2; x2; x1; x1|], [|y1; y1; y2; y2; y1|])

let gcd a b =
    BigInteger.GreatestCommonDivisor (a, b)

let lcm a b =
    (a * b) / gcd a b |> abs


let test1 () =

    let xBox = 7.0
    let yBox = 5.0

    let M = [[0; 0]]
    //let P = [3]

    let M = M |> HostTensor.ofList2D

    use pl = new PLStream()
    pl.sdev "xcairo"
    pl.setopt("geometry", "1280x1024") |> ignore
    pl.init()
    pl.env(-8.0, 8.0, -6.0, 6.0, AxesScale.Independent, AxisBox.BoxTicksLabels)
    pl.lab("x", "y", "")

    // draw lattice
    let x = HostTensor.arange -xBox 1.0 xBox 
    let y = HostTensor.arange -yBox 1.0 yBox 
    let xg = Tensor.replicate 1 y.Shape.[0] x.[*, NewAxis] |> Tensor.flatten |> HostTensor.toArray
    let yg = Tensor.replicate 0 x.Shape.[0] y.[NewAxis, *] |> Tensor.flatten |> HostTensor.toArray
    pl.col0 Color.Grey
    pl.poin (xg, yg, Symbol.Dot)
    rectangle pl (xBox, yBox) (-xBox, -yBox)

    // // calculate matrix inverse
    // let M = M |> HostTensor.ofList2D |> Tensor.convert<Rat>
    // let I, S, N = LinAlg.generalInverse M
    // let _, SN, _ = LinAlg.generalInverse N
    
    // printMat "matrix M" M
    // printMat "inverse I" I
    // printMat "solvability S" S
    // printMat "nullspace N" N
    // printMat "solvability of nullspace SN" SN
    // printfn ""

    // // bring nullspace in integer form by multiplying each row with the LCM of its denominators
    // let LCM = SN |> Tensor.foldAxis (fun l x -> lcm l x.Dnm) (HostTensor.scalar (bigint 1)) 1
    // let SN = SN * (Tensor.convert<Rat> LCM.[*, NewAxis]) |> Tensor.convert<bigint>
    // printMat "integer solvability of nullspace SN" SN

    // // calculate Smith normal form of nullspace
    // let SNU, SNS, SNV = LinAlg.smithNormalForm SN
    // printMat "SNU" SNU
    // printMat "SNS" SNS
    // printMat "SNV" SNV

    // calculate integer inverse of M
    printfn "integer inversion:"
    let I, S, N = LinAlg.integerInverse (Tensor.convert<bigint> M)
    printMat "inverse I" I
    printMat "solvability S" S
    printMat "nullspace N" N   


    // plot nullspace lattice
    let plotNullspacePoints x0 =
        let zMin, zMax = bigint -5, bigint 5
        let rec doPlot (z: Tensor<bigint>) =
            let x = x0 + N .* z |> Tensor.convert<float>
            pl.col0 Color.BlueViolet
            let lbl = z |> Tensor.convert<int> |> HostTensor.toList |> List.map (sprintf "%d") |> String.concat ""
            pl.string2 ([|x.[[0L]]|], [|x.[[1L]]|], lbl)

            let rec incr pos =
                if pos >= z.NElems then false
                else
                    if z.[[pos]] = zMax then 
                        z.[[pos]] <- zMin
                        incr (pos+1L)
                    else
                        z.[[pos]] <- z.[[pos]] + bigint.One
                        true
            if incr 0L then doPlot z                    
        doPlot (HostTensor.filled [N.Shape.[1]] zMin)


    let ys = [[-2]; [-1]; [0]; [1]; [2]] 
    for y in ys do
        let y = HostTensor.ofList y |> Tensor.convert<bigint>        
        printfn "y point: %A" y
        if S .* y ==== bigint.Zero |> Tensor.all then
            printfn "fulfilles solvabilty constraint"
            let x0 = I .* Tensor.convert<Rat> y
            if x0 |> Tensor.map Rat.isInteger |> Tensor.all then
                printfn "produces integer solution"
                let grp = sprintf "%A" y
                pl.schr(0.0, 0.7)
                plotNullspacePoints (Tensor.convert<bigint> x0)
            else
                printfn "does not produce integer solution"
        else
            printfn "does not fulfill solvability contraint"         
    

    //exit 0

    // for p in [0; 1; 2; 3] do
    //     let P = [p]
    //     // invert point P
    //     let P = P |> HostTensor.ofList |> Tensor.convert<Rat>
    //     let R = I .* P

    //     //printMat "point P" P
    //     //printMat "inverse of point R" R

    //     // so we have the inverse point and the nullspace vector

    //     // draw R point
    //     let Rx, Ry = float R.[[0L]], float R.[[1L]]
    //     pl.col0 Color.Red
    //     pl.poin ([|Rx|], [|Ry|], 'x')

    //     // draw base vectors of null-space of M
    //     let f=10.
    //     let Nx, Ny = float N.[[0L; 0L]], float N.[[1L; 0L]]
    //     pl.col0 Color.BlueViolet
    //     pl.line ([|Rx-f*Nx; Rx; Rx+f*Nx|], [|Ry-f*Ny; Ry; Ry+f*Ny|])

    //     // draw orthgonal base vectors of null-space of R
    //     let f=0.5
    //     let SNx, SNy = float SN.[[0L; 0L]], float SN.[[0L; 1L]]
    //     pl.col0 Color.Magenta
    //     pl.line ([|Rx; Rx+f*SNx|], [|Ry; Ry+f*SNy|])        


let test2() =
    let M = [[2; 1; 3];
             [1; 2; 3];
             [3; 3; 6]]
    let M = M |> HostTensor.ofList2D |> Tensor.convert<Rat>
    let A = HostTensor.identity 3L

    let R, _, _, B = LinAlg.rowEchelonAugmented M A

    printMat "M" M
    printMat "A" A
    printMat "row echelon R" R
    printMat "inverse B" B

    // now the row echelon is already integer but the inverse is not

    // calculate the LCM of the denominators of all rows
    //or row in 0L .. M.Shape.[0] - 1L do
    let LCMR = R |> Tensor.foldAxis (fun l x -> lcm l x.Dnm) (HostTensor.scalar (bigint 1)) 1
    let LCMRB = B |> Tensor.foldAxis (fun l x -> lcm l x.Dnm) LCMR 1

    printMat "LCM AL" LCMRB

    // multiply the rows accordingly
    let LCMRB = LCMRB |> Tensor.convert<Rat> |> Tensor.padRight
    let R = R * LCMRB
    let B = B * LCMRB

    printfn "after multiplication with LCM:"
    printMat "row echelon R" R
    printMat "inverse B" B
    
    // next step:
    // understand the meaning of R and B now


let test3() =
    //let M = [[10; 3; 3; 8];
    //         [6; -7; 0;-5]]
    let M = [[0; 0; 3; 8];
             [0; 0; 0;-5]]             
    let M = M |> HostTensor.ofList2D |> Tensor.convert<Rat>

    let I, S, N = LinAlg.generalInverse M
   
    printMat "matrix M" M
    printMat "inverse I" I
    printMat "solvability S" S
    printMat "nullspace N" N

let test4() =
    let M = [[2; 1; 3];
             [1; 2; 3];
             [3; 3; 6]]
    let M = M |> HostTensor.ofList2D |> Tensor.convert<bigint>
    let U, S, V = LinAlg.smithNormalForm M

    printMat "matrix M" M
    printMat "U" U
    printMat "Smith S" S
    printMat "V" V


let setupPlot (pl: PLStream) xRng yRng =
    pl.sdev "xcairo"
    pl.setopt("geometry", "1280x1024") |> ignore
    pl.init()
    pl.env(-xRng, xRng, -yRng, yRng, AxesScale.Independent, AxisBox.BoxTicksLabels)
    pl.lab("x", "y", "")

let drawLattice (pl: PLStream) xRng yRng =
    let x = HostTensor.arange -xRng 1.0 xRng
    let y = HostTensor.arange -yRng 1.0 yRng 
    let xg = Tensor.replicate 1 y.Shape.[0] x.[*, NewAxis] |> Tensor.flatten |> HostTensor.toArray
    let yg = Tensor.replicate 0 x.Shape.[0] y.[NewAxis, *] |> Tensor.flatten |> HostTensor.toArray
    pl.col0 Color.Grey
    pl.poin (xg, yg, Symbol.Dot)    

let drawPolygon (pl: PLStream) (polygon: Tensor<float>) =
    let x = polygon.[0L, *] |> HostTensor.toList
    let y = polygon.[1L, *] |> HostTensor.toList    
    pl.line(x @ [x.Head] |> List.toArray, y @ [y.Head] |> List.toArray)

let testRectTransform() =
    let xRng, yRng = 10., 10.
    use pl = new PLStream()
    setupPlot pl xRng yRng
    drawLattice pl xRng yRng

    // generate rectangle points
    let genBox (xMin, yMin) (xMax, yMax) =
        // upper line
        let ux = [xMin .. xMax]
        let uy = List.replicate ux.Length yMax
        // right line
        let ry = [yMax .. -1 .. yMin]
        let rx = List.replicate ry.Length xMax
        // lower line
        let dx = [xMax .. -1 .. xMin]
        let dy = List.replicate dx.Length yMin
        // left line
        let ly = [yMin .. yMax-1]
        let lx = List.replicate ly.Length xMin

        let x = ux @ rx @ dx @ lx |> HostTensor.ofList
        let y = uy @ ry @ dy @ ly |> HostTensor.ofList
        Tensor.concat 0 [x.[NewAxis, *]; y.[NewAxis, *]]

    let box = genBox (-3, -2) (2, 1)
    pl.col0 (Color.Aquamarine)
    drawPolygon pl (Tensor.convert<float> box)

    // transform box into basis space
    let B = [[1; 0]
             [3;-1]]
    let B = HostTensor.ofList2D B
    let IB, _, _ = LinAlg.integerInverse (Tensor.convert<bigint> B)
    let IB = Tensor.convert<int> IB
    printMat "Basis    B" B
    printMat "Inverse IB" IB
    printMat "IB .* B" (IB .* B)

    let bbox = IB .* box
    pl.col0 (Color.Salmon)
    drawPolygon pl (Tensor.convert<float> bbox)


    printMat "box" box
    printMat "bbox" bbox

    ()



type DimFaith =
    /// Dimension is free with given index
    | Free of int
    /// Dimension is dependant with given index
    | Dependant of int


type SumInfo = {
    // xFree are the summation coordinates.
    // xDep = XFreeToXDep .* XFree + YToXDep .* Y
    // x = [xFree; xDep]

    /// Solvability .* Y = 0 for system to be solvable
    Solvability:        Tensor<bigint>

    /// What has become of a coordinate?
    XDimFaith:          DimFaith list

    /// Matrix mapping from free to dependant coordinates
    XFreeToXDep:       Tensor<Rat>

    /// Matrix mapping from input coordinates to dependant coordinates
    YToXDep:            Tensor<Rat>

    // Constraints 1 are of the form:
    // XFreeToC .* xFree + YToC .* Y >= cb

    /// Matrix mapping from free coordinates to constraints
    XFreeToC1:          Tensor<Rat> 

    /// Matrix mapping from input coordinates to constraints
    YToC1:              Tensor<Rat>

    /// Constraint bias 
    CB1:                Tensor<int64> 

    // Constraints 2 are of the form:
    // XFreeToC .* xFree + YToC .* Y <= cb

    /// Matrix mapping from free coordinates to constraints
    XFreeToC2:          Tensor<Rat> 

    /// Matrix mapping from input coordinates to constraints
    YToC2:              Tensor<Rat>

    /// Constraint bias 
    CB2:                Tensor<int64>     

    // Divisibility constraints are of the form:
    // XFree = YToR .* Y (mod RDiv)

    /// Matrix mapping from input coordinates to division remainders
    YToRDiv:            Tensor<Rat>

    /// Dividers
    RDiv:               Tensor<bigint>       
}


type InputRange = int64 * int64


let doSummation (si: SumInfo) (Y: Tensor<bigint>) =
    // we want to perform a summation and for that we need to know the summation ranges
    // to find the lower summation range what do we do?
    let toRat = Tensor.convert<Rat>

    // build constraint system
    let A1 = si.XFreeToC1
    let b1 = toRat si.CB1 - si.YToC1 .* toRat Y
    let A2 = -si.XFreeToC2
    let b2 = -toRat si.CB2 + si.YToC2 .* toRat Y
    let A = Tensor.concat 0 [A1; A2]
    let b = Tensor.concat 0 [b1; b2]

    printfn "Constraint system: A .* xFree >= b"
    printMat "A" A
    printMat "b" b

    match FourierMotzkin.solve A b with
    | Some sol ->
        let rec doSum sol xFree =
            let j = FourierMotzkin.active sol
            if j >= 0L then
                let low, high = FourierMotzkin.range sol
                let low = low |> ceil |> int
                let high = high |> floor |> int
                for i in low .. high do
                    doSum (sol |> FourierMotzkin.subst (Rat i)) (i :: xFree)
            else
                let xFree = xFree |> HostTensor.ofList
                let xDep = si.XFreeToXDep .* toRat xFree + si.YToXDep .* toRat Y
                printfn "xFree=%A   xDep=%A" xFree xDep
        doSum sol []
    | None ->
        printfn "Summation range is empty."

    ()


let test5() =


    // let M = [[2; 1; 3];
    //          [1; 2; 3];
    //          [3; 3; 6]]
    //let M = [[1; 2; 3];
    //         [2; 4; 6];
    //         [3; 1; 6]]      
    let M = [[1; 2; 3];
             [1; 2; 3];
             [1; 2; 3]]      

    // Y = M .* X

    let XRanges = [(0L, 10L); (0L, 20L); (0L, 50L)]
    let Y = [6; 6; 6] |> HostTensor.ofList |> Tensor.convert<bigint>



    let M = M |> HostTensor.ofList2D |> Tensor.convert<bigint>
    let I, S, N = LinAlg.integerInverse M


    let nOut, nIn = M.Shape.[0], M.Shape.[1]


    printMat "M" M
    printMat "inverse   I" I
    printMat "nullspace N" N

    // so N is the nullspace basis matrix
    let dNull = N.Shape.[1]

    // We need as many input dimensions as there are columns, the remaining dimensions will become dependants.
    // By removing rows with minimum absolute value first, it is ensured that zero rows are always removed.
    let dependants = 
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
        |> List.filter (fun r -> not (List.contains r dependants))

    let getRows rowList (V: Tensor<_>) =
        rowList
        |> List.map (fun r -> V.[int64 r .. int64 r, *])
        |> Tensor.concat 0
    
    let ND = getRows dependants N
    let NF = getRows free N

    let ID = getRows dependants I
    let IF = getRows free I

    printMat "Dependants ND" ND
    printMat "Free       NF" NF

    // Now write the dependants as a function of the free.
    // This is done by solving the free for the basis factors and then inserting into the dependants.
    let NFI, _, _ = LinAlg.integerInverse NF
    let NDF = Tensor.convert<Rat> ND .* NFI

    printMat "Dependants given Free NDF" NDF

    // okay, what needs to be done??
    // add modulo determination
    
    // okay how to determine modulos?
    // dependants must be whole numbers so compute the LCM of all rows

    // computing LCM of NDF
    // Each free coordinate must be divisable by it.
    let FD =
        NDF
        |> Tensor.foldAxis (fun l v -> lcm l v.Dnm) (HostTensor.scalar bigint.One) 1

    printMat "Free divisibiltiy requirements FD" FD



    let dimFaith =
        [0 .. int nIn-1]
        |> List.map (fun d ->
            match free |> List.tryFind ((=) d), dependants |> List.tryFind ((=) d) with
            | Some fd, None -> Free fd
            | None, Some dd -> Dependant dd
            | _ -> failwith "dimension cannot be both free and dependant")


    // now what is the inverse doing?
    // mapping to all coordinates actually
    // 

    let xLow = XRanges |> List.map fst |> HostTensor.ofList
    let xHigh = XRanges |> List.map snd |> HostTensor.ofList

    let XFreeId = HostTensor.identity (int64 free.Length)
    let XFreeZeros = HostTensor.zeros [int64 free.Length; nOut]

    let XFreeConstr = Tensor.concat 0 [XFreeId; NDF]
    let YConstr = Tensor.concat 0 [XFreeZeros; ID - NDF .* IF]

    // just fill out the record
    let si = {
        Solvability = S
        XDimFaith   = dimFaith
        XFreeToXDep = NDF
        YToXDep     = ID - NDF .* IF 
        XFreeToC1   = XFreeConstr
        YToC1       = YConstr
        CB1         = xLow
        XFreeToC2   = XFreeConstr
        YToC2       = YConstr
        CB2         = xHigh
        YToRDiv     = ID
        RDiv        = FD
    }
    

    printfn "SumInfo:\n%A" si

    printMat "Y" Y

    doSummation si Y
        
    ()





let testInequal () =
    let A = [[ 3; 5; 7; 8]
             [-5; 5; 2; 3]
             [ 0; 1; 2; 3]
             [ 1; 2; 3; 4]] |> HostTensor.ofList2D |> Tensor.convert<Rat>
    let b = [2; 3; 4; 0] |> HostTensor.ofList |> Tensor.convert<Rat>

    // let A = [[ 1;  0; 0]
    //          [-1; -2; 0]
    //          [ 1;  1; 0]
    //          [ 1; -1; 0]
    //          [ 0;  1; 0]
    //          [ 2;  1; 1]]
    //         |> HostTensor.ofList2D |> Tensor.convert<Rat>
    // let b = [0; -6; 2; 3; 0; 0] |> HostTensor.ofList |> Tensor.convert<Rat>

    printMat "A" A
    printMat "b" b

    let sol = FourierMotzkin.solve A b  
    match sol with
    | Some sol ->        
        printfn "Solution exists:"
        let rec printSol sol =
            let j = FourierMotzkin.active sol
            if j >= 0L then
                let low, high = FourierMotzkin.range sol
                printfn "%A <= x_%d <= %A" low j high
                let value = 
                    if low > Rat.NegInf then low
                    elif high < Rat.PosInf then high
                    else Rat.Zero
                printfn "Setting x_%d = %A" j value
                printSol (sol |> FourierMotzkin.subst value)
        printSol sol        
    | None ->
        printfn "No solution exists."
    ()



[<EntryPoint>]
let main argv =
    match argv with
    | [|"test1"|] -> test1()
    | [|"test2"|] -> test2()
    | [|"test3"|] -> test3()
    | [|"test4"|] -> test4()
    | [|"testRectTransform"|] -> testRectTransform()
    | [|"test5"|] -> test5()
    | [|"testInequal"|] -> testInequal()
    | _ -> failwith "unknown"
    0


