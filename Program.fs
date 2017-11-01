open Elements

let printAllDerivs fn =       
    let dFns = Elements.derivFunc fn
    let dInArg = "d" + fn.Name
    printfn "Input: %A" fn
    for KeyValue(v, dFn) in dFns do
        printfn "Derivative of %s w.r.t. %s: %A" fn.Name v dFn


let doDemo () =
    let i, iSize = Elements.pos "i", 3L
    let j, jSize = Elements.pos "j", 4L
    let k, kSize = Elements.pos "k", 5L  // summation index

    let dimNames = [i.Name; j.Name]
    let dimSizes = Map [i.Name, iSize; j.Name, jSize]    

    let argShapes = Map ["a",[iSize; kSize]; "b",[jSize; kSize]; "c",[iSize; iSize]; "d",[iSize + kSize]]

    let a, b, c, d = Elements.arg "a", Elements.arg "b", Elements.arg "c", Elements.arg "d"
    let summand = (a[i;k] + b[j;k])**2. * c[i;i] + (d[i+k])**3.
    let expr = exp (- Elements.sumConstRng "k" 0L (kSize-1L) summand)
    let func = Elements.func "f" dimNames dimSizes argShapes expr

    printAllDerivs func         


[<EntryPoint>]
let main argv =
    doDemo ()
    0



