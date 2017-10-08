module ConsumerTests


open System
open System.Numerics
open Tensor
open Tensor.Algorithms


open OldTests


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



let verifyConsumers (m: Tensor<bigint>) (xRngs: Consumers.Range list) =
    
    printMat "M" m
    printfn "x ranges: %A" xRngs

    let n = List.length xRngs

    // Build set for each y, which x hits it.
    let yHitters = System.Collections.Generic.Dictionary()
    let rec doIter rngs x =
        match rngs with
        | (low, high) :: rRngs ->
            for i in low .. high do
                doIter rRngs (x @ [i])
        | [] ->
            let xi = x |> HostTensor.ofList |> Tensor.convert<bigint>
            let y = m .* xi |> Tensor.convert<int> |> HostTensor.toList
            if not (yHitters.ContainsKey y) then
                yHitters.[y] <- System.Collections.Generic.HashSet (Seq.singleton x)
            else
                yHitters.[y].Add x |> ignore
    doIter xRngs []

    // Calculate the range for each dimension of y.
    let ys =
        yHitters.Keys
        |> Seq.map (HostTensor.ofList)
        |> Seq.map (fun y -> y.[NewAxis, *])
        |> Tensor.concat 0
    let yLow = Tensor.minAxis 0 ys
    let yHigh = Tensor.maxAxis 0 ys
    //printfn "yLow:  %A" yLow
    //printfn "yHigh: %A" yHigh

    // Add unhit ys within range for checking.
    let mutable maxUnhit = 100
    let rec addUnhit d y =
        if d >= 0L then
            for i in (yLow.[[d]] - 5) .. (yHigh.[[d]] + 5) do
                addUnhit (d-1L) (i::y)
        else
            if not (yHitters.ContainsKey y) && maxUnhit > 0 then
                //printfn "adding unhit y=%A" y
                yHitters.[y] <- System.Collections.Generic.HashSet()
                maxUnhit <- maxUnhit - 1
    addUnhit (yLow.NElems - 1L) []

    // Build consumer info.
    let ci = Consumers.compute m xRngs
    //printfn "ci:\n%A" ci

    // compare
    for KeyValue(y, xs) in yHitters do
        let xs = Set.ofSeq xs
        let xsComp = Consumers.get ci (y |> HostTensor.ofList |> Tensor.convert<bigint>) |> Set.ofSeq
        if xs = xsComp then
            //printfn "For y=%A:" y
            //printfn "Really hitting x:   %A" (xs |> Set.toList)
            //printfn "Computed hitting x: %A" (xsComp |> Set.toList)        
            //printfn "Match!"
            //printfn ""
            ()
        else
            printfn "For y=%A:" y
            printfn "Really hitting x:   %A" (xs |> Set.toList)
            printfn "Computed hitting x: %A" (xsComp |> Set.toList)        
            printfn "===== Mismatch!!! ====="
            exit 1
    printfn "verifyConsumers okay!"


let testConsumers2() =
    let doVerify m xRngs =
        verifyConsumers (m |> HostTensor.ofList2D |> Tensor.convert<bigint>) xRngs     

    let M = [[1; 2; 3];
             [1; 2; 3];
             [1; 2; 3]]     
    let t1 = async { doVerify M [(0L, 10L); (0L, 20L); (0L, 50L)] }

    let M = [[1; 2; 3];
             [2; 4; 6];
             [3; 1; 6]]      
    let t2 = async { doVerify M [(0L, 10L); (0L, 20L); (0L, 50L)] }

    let M = [[2; 1; 3];
             [1; 2; 3];
             [5; 3; 6]]
    let t3 = async { doVerify M [(0L, 10L); (0L, 20L); (0L, 50L)] }

    let M = [[1]]
    let t4 = async { doVerify M [(0L, 10L);] }

    let M = [[-5]]
    let t5 = async { doVerify M [(0L, 15L);] }

    let M = [[0]]
    let t6 = async { doVerify M [(0L, 15L);] }

    let M = [[2];
             [3]]
    let t7 = async { doVerify M [(0L, 15L);] }

    let M = [[2; 5];
             [3; 4]]
    let t8 = async { doVerify M [(0L, 15L); (0L, 12L);] }

    let M = [[0; 0; 1];
             [1; 0; 0];
             [0; 1; 0]]
    let t9 = async { doVerify M [(0L, 10L); (0L, 5L); (0L, 7L)] }

    let M = [[0; 0; 0];
             [0; 0; 0];
             [0; 0; 0]]
    let t10 = async { doVerify M [(0L, 10L); (0L, 5L); (0L, 7L)] }

    let M = [[1; 2; 4; 2];
             [2; 4; 1; 3];
             [3; 6; 2; 4];
             [4; 8; 3; 4]]
    let t11 = async { doVerify M [(0L, 10L); (0L, 5L); (0L, 7L); (0L, 8L)] }

    let M = [[0; 0; 0; 0];
             [0; 0; 0; 0];
             [0; 0; 0; 0];
             [4; 7; 3; 5]]
    let t12 = async { doVerify M [(0L, 10L); (0L, 5L); (0L, 7L); (0L, 8L)] }    

    Async.Parallel [t1; t2; t3; t4; t5; t6; t7; t8; t9; t10; t11; t12] |> Async.RunSynchronously |> ignore


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

