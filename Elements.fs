namespace Elements

open Tensor
open System


/// element expression
module Elements =

    /// An expression for an index as a linear combination.
    [<StructuredFormatDisplay("{Pretty}")>]
    type IdxExpr = 
        IdxExpr of Map<string, int64>
        with
            static member factor dim value =
                IdxExpr (Map [dim, value])
            static member (~-) (IdxExpr af) =
                af |> Map.map (fun ai av -> -av) |> IdxExpr
            static member (+) (IdxExpr af, IdxExpr bf) =
                let f = bf |> Map.fold (fun f i bv -> match f |> Map.tryFind i with
                                                      | Some v -> f |> Map.add i (v+bv)
                                                      | None -> f |> Map.add i bv) af
                IdxExpr f
            static member (-) (a: IdxExpr, b: IdxExpr) =
                a + (-b)
            static member (*) (f: int64, IdxExpr bf) =
                bf |> Map.map (fun bi bv -> f * bv) |> IdxExpr
            static member (/) (IdxExpr af, f: int64) =
                af |> Map.map (fun ai av -> av / f) |> IdxExpr
            member this.Pretty =
                let (IdxExpr f) = this
                Map.toList f
                |> List.map fst
                |> List.sort
                |> List.choose (fun n -> 
                    if f.[n] = 0L then None
                    elif f.[n] = 1L  then Some n
                    elif f.[n] = -1L then Some ("-" + n)
                    else Some (sprintf "%d*%s" f.[n] n))
                |> String.concat " + "
            static member name (IdxExpr f) =
                f |> Map.toList |> List.exactlyOne |> fst

    /// Index expressions for all indicies of a tensor.
    [<StructuredFormatDisplay("{Pretty}")>]    
    type IdxExprs =
        IdxExprs of IdxExpr list
        with
            static member toMatrix inNames (IdxExprs idx) =
                let nIn = List.length inNames |> int64
                let nOut = idx |> List.length |> int64
                let m = HostTensor.zeros [nOut; nIn]
                idx |> List.iteri (fun r (IdxExpr f) ->
                    f |> Map.iter (fun name v -> 
                        match inNames |> List.tryFindIndex ((=) name) with
                        | Some c -> m.[[int64 r; int64 c]] <- v
                        | None -> failwithf "dimension %s does not exist" name))
                m          
            member this.Pretty =
                let (IdxExprs idx) = this
                sprintf "%A" idx

    type LeafOp =
        | Const of float
        | IdxValue of idx:IdxExprs
        | Argument of name:string * idx:IdxExprs

    and UnaryOp = 
        | Negate                        
        | Abs
        | Sgn
        | Log
        | Log10                           
        | Exp                           
        | Tanh
        | Sqrt
        //| Sum of SizeSymbolT * SizeSpecT * SizeSpecT
        //| KroneckerRng of SizeSpecT * SizeSpecT * SizeSpecT

    and BinaryOp = 
        | Add                           
        | Substract                     
        | Multiply                      
        | Divide                        
        | Modulo
        | Power        
        //| IfThenElse of SizeSpecT * SizeSpecT
        
    /// an element expression
    and [<StructuredFormatDisplay("{Pretty}")>]
        ElemExpr =
        | Leaf of LeafOp
        | Unary of UnaryOp * ElemExpr
        | Binary of BinaryOp * ElemExpr * ElemExpr

    and Dim = string * int64

    and [<StructuredFormatDisplay("{Pretty}")>]
        ElemFunc = {
            Expr:       ElemExpr
            Shape:      Dim list
        } with
            member this.Pretty =
                let dims =
                    this.Shape
                    |> List.map fst
                    |> String.concat "; "
                sprintf "f[%s] = %A" dims this.Expr

    let func dims expr =
        {Expr=expr; Shape=dims |> List.map (fun (i, s) -> IdxExpr.name i, s)}

    /// a constant value given by a ConstSpec
    let scalar v = Leaf (Const v) 
         
    type ElemExpr with

        // elementwise unary
        static member (~+) (a: ElemExpr) = a 
        static member (~-) (a: ElemExpr) = Unary(Negate, a) 
        static member Abs (a: ElemExpr) = Unary(Abs, a) 
        static member Sgn (a: ElemExpr) = Unary(Sgn, a) 
        static member Log (a: ElemExpr) = Unary(Log, a) 
        static member Log10 (a: ElemExpr) = Unary(Log10, a) 
        static member Exp (a: ElemExpr) = Unary(Exp, a) 
        static member Tanh (a: ElemExpr) = Unary(Tanh, a) 
        static member Sqrt (a: ElemExpr) = Unary(Sqrt, a) 

        // elementwise binary
        static member (+) (a: ElemExpr, b: ElemExpr) = Binary(Add, a, b) 
        static member (-) (a: ElemExpr, b: ElemExpr) = Binary(Substract, a, b) 
        static member (*) (a: ElemExpr, b: ElemExpr) = Binary(Multiply, a, b) 
        static member (/) (a: ElemExpr, b: ElemExpr) = Binary(Divide, a, b) 
        static member (%) (a: ElemExpr, b: ElemExpr) = Binary(Modulo, a, b) 
        static member Pow (a: ElemExpr, b: ElemExpr) = Binary(Power, a, b) 
        static member ( *** ) (a: ElemExpr, b: ElemExpr) = a ** b 

        // elementwise binary with basetype
        static member (+) (a: ElemExpr, b: float) = a + (scalar b) 
        static member (-) (a: ElemExpr, b: float) = a - (scalar b) 
        static member (*) (a: ElemExpr, b: float) = a * (scalar b) 
        static member (/) (a: ElemExpr, b: float) = a / (scalar b) 
        static member (%) (a: ElemExpr, b: float) = a % (scalar b) 
        static member Pow (a: ElemExpr, b: float) = a ** (scalar b) 
        static member ( *** ) (a: ElemExpr, b: float) = a ** (scalar b)

        static member (+) (a: float, b: ElemExpr) = (scalar a) + b 
        static member (-) (a: float, b: ElemExpr) = (scalar a) - b 
        static member (*) (a: float, b: ElemExpr) = (scalar a) * b 
        static member (/) (a: float, b: ElemExpr) = (scalar a) / b 
        static member (%) (a: float, b: ElemExpr) = (scalar a) % b 
        static member Pow (a: float, b: ElemExpr) = (scalar a) ** b   
        static member ( *** ) (a: float, b: ElemExpr) = (scalar a) ** b          

        member private this.PrettyAndPriority = 
            match this with
            | Leaf (op) -> 
                let myPri = 20
                let myStr =
                    match op with
                    | Const v -> sprintf "%g" v
                    | IdxValue idx -> sprintf "(%A)" idx
                    | Argument (name, idxs) -> sprintf "%s%A" name idxs
                myStr, myPri
            
            | Unary (op, a) ->
                let myPri = 10
                let aStr, aPri = a.PrettyAndPriority
                let aStr =
                    if myPri > aPri then sprintf "(%s)" aStr
                    else aStr
                let myStr = 
                    match op with
                    | Negate -> sprintf "(-%s)" aStr
                    | Abs -> sprintf "abs %s" aStr
                    | Sgn -> sprintf "sgn %s" aStr
                    | Log -> sprintf "log %s" aStr
                    | Log10 -> sprintf "log10 %s" aStr
                    | Exp -> sprintf "exp %s" aStr
                    | Tanh -> sprintf "tanh %s" aStr
                    | Sqrt -> sprintf "sqrt %s" aStr
                myStr, myPri
                
            | Binary(op, a, b) -> 
                let mySym, myPri =
                    match op with
                    | Add -> "+", 1
                    | Substract -> "_", 1
                    | Multiply -> "*", 2
                    | Divide -> "/", 2
                    | Modulo -> "%", 2
                    | Power -> "**", 5
                let aStr, aPri = a.PrettyAndPriority
                let bStr, bPri = b.PrettyAndPriority
                let aStr =
                    if myPri > aPri then sprintf "(%s)" aStr
                    else aStr
                let bStr =
                    if myPri > bPri then sprintf "(%s)" bStr
                    else bStr
                let myStr = sprintf "%s %s %s" aStr mySym bStr
                myStr, myPri            

        member this.Pretty = this.PrettyAndPriority |> fst

    /// sign keeping type
    let sgn (a: ElemExpr) =
        ElemExpr.Sgn a 

    /// square root
    let sqrtt (a: ElemExpr) =
        ElemExpr.Sqrt a 
                  
    /// index symbol for given dimension of the result
    let idxValue idx =
        Leaf (IdxValue idx)           

    /// specifed element of argument 
    let arg name idx =
        Leaf (Argument (name, IdxExprs idx))

    /// index of given name
    let idx name = IdxExpr.factor name 1L

    /// constant index value
    let idxConst v = IdxExpr.factor "1" v

    
    // /// checks if the arguments' shapes are compatible with the result shape and that the types match
    // let checkCompatibility (expr: ElemExpr) (argShapes: ShapeSpecT list) (argTypes: TypeNameT list) 
    //         (resShape: ShapeSpecT) =

    //     // check number of arguments
    //     let nArgs = List.length argShapes
    //     if argTypes.Length <> nArgs then
    //         failwith "argShapes and argTypes must be of same length"
    //     let nReqArgs = requiredNumberOfArgs expr       
    //     if nReqArgs > nArgs then
    //         failwithf "the element expression requires at least %d arguments but only %d arguments were specified"
    //             nReqArgs nArgs

    //     // check dimensionality of arguments
    //     let rec check expr =
    //         match expr with
    //         | Leaf (ArgElement ((Arg n, idx), tn)) ->
    //             if not (0 <= n && n < nArgs) then
    //                 failwithf "the argument with zero-based index %d used in the element \
    //                            expression does not exist" n
    //             let idxDim = ShapeSpec.nDim idx
    //             let argDim = ShapeSpec.nDim argShapes.[n]
    //             if idxDim <> argDim then
    //                 failwithf 
    //                     "the argument with zero-based index %d has %d dimensions but was used  \
    //                      with %d dimensions in the element expression" n argDim idxDim
    //             let argType = argTypes.[n]
    //             if argType <> tn then
    //                 failwithf 
    //                     "the argument with zero-based index %d has type %A but was used  \
    //                      as type %A in the element expression" n argType.Type tn.Type
    //         | Leaf _ -> ()

    //         | Unary (_, a) -> check a
    //         | Binary (_, a, b) -> check a; check b
    //     check expr

    // /// substitutes the specified size symbols with their replacements 
    // let rec substSymSizes symSizes expr = 
    //     let sSub = substSymSizes symSizes
    //     let sSize = SymSizeEnv.subst symSizes
    //     let sShp = SymSizeEnv.substShape symSizes

    //     match expr with
    //     | Leaf (SizeValue (sc, tn)) -> Leaf (SizeValue ((sSize sc), tn))
    //     | Leaf (ArgElement ((arg, argIdxs), tn)) -> Leaf (ArgElement ((arg, sShp argIdxs), tn))
    //     | Leaf _ -> expr

    //     | Unary (Sum (sym, first, last), a) -> 
    //         Unary (Sum (sym, sSize first, sSize last), sSub a)
    //     | Unary (KroneckerRng (sym, first, last), a) ->
    //         Unary (KroneckerRng (sSize sym, sSize first, sSize last), sSub a)
    //     | Unary (op, a) -> Unary (op, sSub a)
    //     | Binary (IfThenElse (left, right), a, b) ->
    //         Binary (IfThenElse (sSize left, sSize right), sSub a, sSub b)
    //     | Binary (op, a, b) -> Binary (op, sSub a, sSub b)
