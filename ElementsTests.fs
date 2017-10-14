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
            if not (Tensor.almostEqualWithTol (nJac, aJac, 1e-3, 1e-3)) then
                printfn "Analytic Jacobian:\n%A" aJac
                printfn "Numeric Jacobian:\n%A" nJac
                printfn "Jacobian mismatch!!"
                failwith "Jacobian mismatch in function derivative check"
            else
                //printfn "Analytic Jacobian:\n%A" aJac
                //printfn "Numeric Jacobian:\n%A" nJac            
                //printfn "Analytic and numeric Jacobians match."
                ()

    let randomDerivCheck iters (fn: Elements.ElemFunc) =
        let rnd = System.Random 123
        for i in 1 .. iters do
            let argEnv =
                fn.ArgShapes
                |> Map.map (fun _ shp -> 
                    rnd.SeqDouble() |> Seq.map (fun r -> 2. * r - 1.) |> HostTensor.ofSeqWithShape shp)
            checkFuncDerivs argEnv fn            

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

        let rnd = System.Random 123
        let xv = rnd.SeqDouble() |> HostTensor.ofSeqWithShape [iSize; jSize]
        let yv = rnd.SeqDouble() |> HostTensor.ofSeqWithShape [jSize; jSize] 
        let zv = rnd.SeqDouble() |> HostTensor.ofSeqWithShape [kSize] 

        let dimNames = [i.Name; j.Name; k.Name]
        let dimSizes = Map [i.Name, iSize; j.Name, jSize; k.Name, kSize]    
        let argShapes = Map ["x", xv.Shape; "y", yv.Shape; "z", zv.Shape]

        let expr = Elements.arg "x" [i; j] ** 2.0 + 2.0 * (Elements.arg "y" [j; j] * (Elements.arg "z" [k])**3.0)
        let func = Elements.func "f" dimNames dimSizes argShapes expr

        printfn "x=\n%A" xv
        printfn "y=\n%A" yv
        printfn "z=\n%A" zv
        let argEnv = Map ["x", xv; "y", yv; "z", zv]
        let fv = Elements.evalFunc argEnv func
        checkFuncDerivs argEnv func


            
    [<Fact>]
    let ``DerivCheck2`` () =
        let i, iSize = Elements.pos "i", 3L
        let j, jSize = Elements.pos "j", 4L
        let k, kSize = Elements.pos "k", 5L

        let dimNames = [i.Name; j.Name; k.Name]
        let dimSizes = Map [i.Name, iSize; j.Name, jSize; k.Name, kSize]    
        let argShapes = Map ["x", [iSize; jSize]; "y", [jSize; jSize]; "z", [kSize]]

        let expr = Elements.arg "x" [i; j] ** 2.0 + 2.0 * (Elements.arg "y" [j; j] * (Elements.arg "z" [k])**3.0)
        let func = Elements.func "f" dimNames dimSizes argShapes expr

        randomDerivCheck 10 func


    [<Fact>]
    let ``DerivCheck3`` () =
        let r, rSize = Elements.pos "r", 2L
        let s, sSize = Elements.pos "s", 3L
        let n, nSize = Elements.pos "n", 4L

        let dimNames = [r.Name; s.Name; n.Name]
        let dimSizes = Map [r.Name, rSize; s.Name, sSize; n.Name, nSize]    
        let argShapes = Map ["Sigma", [sSize; nSize; nSize]; "mu", [sSize; nSize]; "V", [rSize; nSize]]

        let Sigma = Elements.arg "Sigma"
        let mu = Elements.arg "mu"
        let V = Elements.arg "V"
        let expr =  // added **2 to Sigma to make it positive
            sqrt (1. / (1. + 2. * Sigma[s;n;n]**2.)) * exp (- (mu[s;n] - V[r;n])**2. / (1. + 2. * Sigma[s;n;n]))
        let func = Elements.func "S" dimNames dimSizes argShapes expr

        randomDerivCheck 10 func

    [<Fact>]
    let ``DerivCheck4`` () =
        let r, rSize = Elements.pos "r", 2L
        let s, sSize = Elements.pos "s", 3L
        let t, sSize = Elements.pos "s", 2L        // =r
        let n, nSize = Elements.pos "n", 4L

        let dimNames = [r.Name; s.Name; n.Name]
        let dimSizes = Map [r.Name, rSize; s.Name, sSize; n.Name, nSize]    
        let argShapes = Map ["Sigma", [sSize; nSize; nSize]; "mu", [sSize; nSize]; "V", [rSize; nSize]]

        let Sigma = Elements.arg "Sigma"
        let mu = Elements.arg "mu"
        let V = Elements.arg "V"
        let expr =  // added **2 to Sigma to make it positive
            sqrt (1. / (1. + 4. * Sigma[s;n;n]**2.)) * exp (- 2. * (mu[s;n] - (V[r;n] + V[t;n])/2.)**2. / (1. + 4. * Sigma[s;n;n]) - 
                (V[r;n] - V[t;n])**2. / 2.)
        let func = Elements.func "S" dimNames dimSizes argShapes expr

        randomDerivCheck 10 func        