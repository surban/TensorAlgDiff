
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




let testConsumers() =

    let xRngs = [(0L, 10L); (0L, 20L); (0L, 50L)]

    // let M = [[2; 1; 3];
    //          [1; 2; 3];
    //          [3; 3; 6]]
    // let M = [[1; 2; 3];
    //         [2; 4; 6];
    //         [3; 1; 6]]      
    let M = [[1; 2; 3];
             [1; 2; 3];
             [1; 2; 3]]      

    let y = [20; 6; 6] 


    // Y = M .* X
    let M = M |> HostTensor.ofList2D |> Tensor.convert<bigint>
    let y = y |> HostTensor.ofList |> Tensor.convert<bigint>
    let ci = Consumers.compute M xRngs
    //printfn "ConsumerInfo:\n%A" ci
    let xs = Consumers.get ci y |> List.ofSeq   

    printfn "y = M .* x"
    printMat "M" M
    printfn "All x for y=%A within range %A:\n%A" y xRngs xs



let testConsumers2() =
    // How to do testing?
    // Basically we must ensure that the summation hits all x points that hit a particular y point.
    // So, we need to build a list of x points hitting the y point and check if it is correct.
    // So, first we need to iterate over the x grid within its limits.
    // Since it is rectangular, this is easy.
    // What then about the y grid we are testing on?
    // => For now only check y that have been hit.
    // What to do if x hits a fractional y?
    // => ignore it for now
    // okay, easy to do
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

    let sol = FourierMotzkin.solve (FourierMotzkin.presolve A) b  
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
    | [|"testConsumers"|] -> testConsumers()
    | [|"testConsumers2"|] -> testConsumers2()
    | [|"testInequal"|] -> testInequal()
    | _ -> failwith "unknown"
    0


