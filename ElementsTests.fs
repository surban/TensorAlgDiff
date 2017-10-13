namespace Elements

open Xunit
open Xunit.Abstractions
open FsUnit.Xunit

open Tensor


module DerivCheck =

    /// evaluates the Jacobian of f at x numerically with specified finite difference step
    let inline numDerivEpsilon (epsilon: 'T) (f: Tensor<'T> -> Tensor<'T>) (x: Tensor<'T>) =
        let y = f x
        let xElems, yElems = Tensor.nElems x, Tensor.nElems y
        let xShp = Tensor.shape x

        let jac = Tensor.zeros x.Dev [yElems; xElems] 
        let xd = x |> Tensor.reshape [xElems] |> Tensor.copy
        for xi in 0L .. xElems-1L do
            let xiVal = xd.[[xi]]
            // f (x+epsilon)
            xd.[[xi]] <- xiVal + epsilon
            let ydf = xd |> Tensor.reshape xShp |> f |> Tensor.reshape [yElems]
            // f (x-epsilon)
            xd.[[xi]] <- xiVal - epsilon
            let ydb = xd |> Tensor.reshape xShp |> f |> Tensor.reshape [yElems]
            // [f (x+epsilon) - f (x-epsilon)] / (2 * epsilon) 
            jac.[*, xi] <- (ydf - ydb) / (Tensor.scalar ydf.Dev (epsilon + epsilon))
            xd.[[xi]] <- xiVal
        jac 

    /// evaluates the Jacobian of f at x numerically
    let numDeriv f x = 
        numDerivEpsilon 1e-5 f x

    let numDerivOfFunc argEnv (fn: Elements.ElemFunc) xName =
        let f xv = Elements.evalFunc (argEnv |> Map.add xName xv) fn
        numDeriv f argEnv.[xName]

    /// Calculates the Jacobian using the derivative of a function.
    let jacobianOfDerivFunc argEnv dInArg (dFn: Elements.ElemFunc) =
        let outElems = dFn.Shape |> List.fold (*) 1L
        let inElems = dFn.ArgShapes.[dInArg] |> List.fold (*) 1L
        let jac = HostTensor.zeros [inElems; outElems]
        for i in 0L .. inElems-1L do
            let dIn = HostTensor.zeros [inElems]
            dIn.[[i]] <- 1.0
            let dIn = dIn |> Tensor.reshape dFn.ArgShapes.[dInArg]
            let dArgEnv = argEnv |> Map.add dInArg dIn
            let dOut = Elements.evalFunc dArgEnv dFn
            jac.[i, *] <- Tensor.flatten dOut
        jac




type ElementsTests (output: ITestOutputHelper) =

    let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 

    let checkFuncDerivs argEnv fn =
        let dFns = Elements.derivFunc fn
        let dInArg = "d" + fn.Name
        printfn "Checking derivative of: %A" fn
        for KeyValue(v, dFn) in dFns do
            printfn "Derivative w.r.t. %s: %A" v dFn
            let aJac = DerivCheck.jacobianOfDerivFunc argEnv dInArg dFn
            let nJac = DerivCheck.numDerivOfFunc argEnv fn v
            //printfn "Analytic Jacobian:\n%A" aJac
            //printfn "Numeric Jacobian:\n%A" nJac
            if not (Tensor.almostEqual nJac aJac) then
                printfn "Jacobian mismatch!!"
                printfn "Analytic Jacobian:\n%A" aJac
                printfn "Numeric Jacobian:\n%A" nJac
                failwith "Jacobian mismatch in function derivative check"
            else
                printfn "Analytic and numeric Jacobians match."


    [<Fact>]
    let ``EvalTest1`` () =
        let i, iSize = Elements.pos "i", 3L
        let j, jSize = Elements.pos "j", 4L
        let k, kSize = Elements.pos "k", 5L

        let xv = HostTensor.zeros [iSize; jSize] + 1.0
        let yv = HostTensor.zeros [jSize; jSize] + 2.0
        let zv = HostTensor.zeros [kSize] + 3.0

        let dimNames = [i.Name; j.Name; k.Name]
        let dimSizes = Map [i.Name, iSize; j.Name, jSize; k.Name, kSize]    
        let argShapes = Map ["x", xv.Shape; "y", yv.Shape; "z", zv.Shape]

        let expr = Elements.arg "x" [i; j] + 2.0 * (Elements.arg "y" [j; j] * (Elements.arg "z" [k])**3.0)
        let func = Elements.func "f" dimNames dimSizes argShapes expr
      
        printfn "Evaluating:"
        printfn "x=\n%A" xv
        printfn "y=\n%A" yv
        printfn "z=\n%A" zv
        let argEnv = Map ["x", xv; "y", yv; "z", zv]
        let fv = Elements.evalFunc argEnv func
        printfn "f=\n%A" fv
                

    [<Fact>]
    let ``DerivTest1`` () =
        let i, iSize = Elements.pos "i", 3L
        let j, jSize = Elements.pos "j", 4L
        let k, kSize = Elements.pos "k", 5L

        let xv = HostTensor.zeros [iSize; jSize] + 1.0
        let yv = HostTensor.zeros [jSize; jSize] + 2.0
        let zv = HostTensor.zeros [kSize] + 3.0

        let dimNames = [i.Name; j.Name; k.Name]
        let dimSizes = Map [i.Name, iSize; j.Name, jSize; k.Name, kSize]    
        let argShapes = Map ["x", xv.Shape; "y", yv.Shape; "z", zv.Shape]

        let expr = Elements.arg "x" [i; j] + 2.0 * (Elements.arg "y" [j; j] * (Elements.arg "z" [k])**3.0)
        let func = Elements.func "f" dimNames dimSizes argShapes expr

        printfn "%A" func
        printfn "Ranges: %A" dimSizes    
        let dFns = Elements.derivFunc func
        printfn "dFns:" 
        for KeyValue(_, dFn) in dFns do
            printfn "%A" dFn


    [<Fact>]
    let ``DerivCheck1`` () =
        let i, iSize = Elements.pos "i", 3L
        let j, jSize = Elements.pos "j", 4L
        let k, kSize = Elements.pos "k", 5L

        let xv = HostTensor.zeros [iSize; jSize] + 1.0
        let yv = HostTensor.zeros [jSize; jSize] + 2.0
        let zv = HostTensor.zeros [kSize] + 3.0

        let dimNames = [i.Name; j.Name; k.Name]
        let dimSizes = Map [i.Name, iSize; j.Name, jSize; k.Name, kSize]    
        let argShapes = Map ["x", xv.Shape; "y", yv.Shape; "z", zv.Shape]

        let expr = Elements.arg "x" [i; j] + 2.0 * (Elements.arg "y" [j; j] * (Elements.arg "z" [k])**3.0)
        let func = Elements.func "f" dimNames dimSizes argShapes expr

        printfn "x=\n%A" xv
        printfn "y=\n%A" yv
        printfn "z=\n%A" zv
        let argEnv = Map ["x", xv; "y", yv; "z", zv]
        let fv = Elements.evalFunc argEnv func
        checkFuncDerivs argEnv func


            

