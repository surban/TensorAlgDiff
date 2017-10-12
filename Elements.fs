namespace Elements

open Tensor
open System


/// element expression
module Elements =

    /// An expression for an index as a linear combination.
    [<StructuredFormatDisplay("{Pretty}")>]
    type IdxExpr = 
        IdxExpr of Map<string, Rat>
        with
            static member zero =
                IdxExpr Map.empty
            static member one =
                IdxExpr.factor "1" Rat.One
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
            static member (*) (f: Rat, IdxExpr bf) =
                bf |> Map.map (fun bi bv -> f * bv) |> IdxExpr
            static member (/) (IdxExpr af, f: Rat) =
                af |> Map.map (fun ai av -> av / f) |> IdxExpr
            member this.Pretty =
                let (IdxExpr f) = this
                let sf =
                    Map.toList f
                    |> List.map fst
                    |> List.sort
                    |> List.choose (fun n -> 
                        if f.[n] = Rat.Zero then None
                        elif f.[n] = Rat.One then Some n
                        elif f.[n] = Rat.MinusOne then Some ("-" + n)
                        elif n = "1" then Some (sprintf "%A" f.[n])
                        else Some (sprintf "%A*%s" f.[n] n))
                if List.isEmpty sf then "0" else sf |> String.concat " + "
            static member name (IdxExpr f) =
                f |> Map.toList |> List.exactlyOne |> fst
            member this.Name = IdxExpr.name this
            static member eval idxEnv (IdxExpr f) =
                let idxEnv = idxEnv |> Map.add "1" Rat.One
                f |> Map.fold (fun s i v -> s + v * idxEnv.[i]) Rat.Zero
            static member subst (repl: Map<string, IdxExpr>) (IdxExpr f) =
                f |> Map.fold (fun r i v -> r + v * repl.[i]) IdxExpr.zero

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
            static member eval idxEnv (IdxExprs idx) =
                idx |> List.map (IdxExpr.eval idxEnv)
            static member subst repl (IdxExprs idx) =
                idx |> List.map (IdxExpr.subst repl) |> IdxExprs
            static member length (IdxExprs idx) =
                List.length idx

    type LeafOp =
        | Const of float
        | IdxValue of idx:IdxExpr
        | Argument of name:string * idxs:IdxExprs

    // Okay, so how to add the nice summation?
    // the nice summation is determined by the ConsumerInfo
    // The ConsumerInfo tells us that we have to sum over the Nullspace.
    // so first we have to see how a summation is implemented as an operation
    // There are several possibilities for that.
    // How to define the sum?
    // 1. a summation takes place over one variable and we nest the sums
    // 2. the sum needs to know its ranges which are specified by the constraints
    // 3. these constraint may depend on already defined indices
    // 4. multiple constraints may exist and act as an AND relation
    // Constraint system for a general sum.
    // L .* s >= R .* i
    // where s are the summation indices.
    // Hmmm, but this doesn't make sense for what we have in ConsumerInfo.
    // There we have a whole system...
    // Yes, there its describing multiple sums, but probably it can be factorized from the
    // Fourier-Motzkin presolution.
    // So the Foruier-Motzkin is, after elimination, giving a set of constraints for the last variable only.
    // But then it the backsubstitution step, its using these values.
    // Yes.
    // So theoretically they could be brought to the other side and just become independant indices, i.e. is.
    // Then the substitution would be done in the matrix multiplication with R.
    // So next step:
    // - figure out how to replace backsubstitution of Fourer-Motzkin by right-side multiplication


    and UnaryOp = 
        | Negate                        
        | Abs
        | Sgn
        | Log
        | Log10                           
        | Exp                           
        | Tanh
        | Sqrt
        | Sum of idx:string * lows:IdxExpr list * highs:IdxExpr list
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

    and [<StructuredFormatDisplay("{Pretty}")>]
        ElemFunc = {
            Name:       string
            DimName:    string list
            DimSize:    Map<string, int64>
            Expr:       ElemExpr
            ArgShape:   Map<string, int64 list>
        } with
            member this.Pretty =
                let dims = this.DimName |> String.concat "; "
                sprintf "%s[%s] = %A" this.Name dims this.Expr
            member this.Shape = 
                this.DimName |> List.map (fun d -> this.DimSize.[d])

    let rec extractArgs expr =
        match expr with
        | Leaf (Argument (name, idxs)) -> Set [name, idxs]
        | Leaf _ -> Set.empty
        | Unary (_, a) -> extractArgs a
        | Binary (_, a, b) -> Set.union (extractArgs a) (extractArgs b)

    let func name dimNames dimSizes argShapes expr =
        for (argName, argIdx) in extractArgs expr do
            match argShapes |> Map.tryFind argName with
            | Some shp when IdxExprs.length argIdx <> List.length shp -> 
                failwithf "shape dimensionality mismatch for argument %s" argName
            | Some shp -> ()
            | None -> failwithf "no shape specified for argument %s" argName                
        {Name=name; DimName=dimNames; DimSize=dimSizes; Expr=expr; ArgShape=argShapes}

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
                    | Sum (sym, lows, highs) -> sprintf "sum:%s_%A^%A (%s)" sym lows highs aStr
                myStr, myPri
                
            | Binary(op, a, b) -> 
                let mySym, myPri =
                    match op with
                    | Add -> "+", 1
                    | Substract -> "-", 1
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
    let pos name = IdxExpr.factor name Rat.One

    /// constant index value
    let idxConst v = IdxExpr.factor "1" v

    let sum idx lows highs a =
        Unary (Sum (idx, lows, highs), a)

    /// substitutes the specified size symbols with their replacements 
    let rec substIdx repl expr = 
        let sub = substIdx repl
        match expr with
        | Leaf (IdxValue idx) -> Leaf (IdxValue (IdxExpr.subst repl idx))
        | Leaf (Argument (name, idxs)) -> Leaf (Argument (name, IdxExprs.subst repl idxs))
        | Leaf (op) -> Leaf (op)
        | Unary (op, a) -> Unary (op, sub a)
        | Binary (op, a, b) -> Binary (op, sub a, sub b)

    let rec evalExpr (argEnv: Map<string, Tensor<float>>) idxEnv expr =
        let subEval = evalExpr argEnv idxEnv
        match expr with
        | Leaf op ->
            match op with
            | Const v -> v
            | IdxValue idx -> idx |> IdxExpr.eval idxEnv |> float
            | Argument (name, idxs) -> 
                let idxs = idxs |> IdxExprs.eval idxEnv |> List.map int64
                let arg = argEnv.[name]
                arg.[idxs]

        | Unary (op, a) ->
            let av = subEval a
            match op with
            | Negate -> -av 
            | Abs -> abs av 
            | Sgn -> Operators.sgn av
            | Log -> log av
            | Log10 -> log10 av
            | Exp -> exp av
            | Tanh -> tanh av
            | Sqrt -> sqrt av

        | Binary (op, a, b) ->
            let av, bv = subEval a, subEval b
            match op with
            | Add -> av + bv
            | Substract -> av - bv                
            | Multiply -> av * bv          
            | Divide -> av / bv           
            | Modulo -> av % bv
            | Power -> av ** bv

    let evalFunc argEnv (func: ElemFunc) =
        let fv = HostTensor.zeros func.Shape
        for pos in TensorLayout.allIdxOfShape func.Shape do
            //printfn "pos is %A" pos
            let idxEnv =
                List.zip pos func.DimName
                |> List.fold (fun env (p, name) -> env |> Map.add name (Rat p)) Map.empty
            fv.[pos] <- evalExpr argEnv idxEnv func.Expr
        fv

    let rec derivExpr expr dExpr =            
        let d = dExpr        
        let rds = derivExpr
        match expr with
        | Leaf op ->
            match op with
            | Const v -> Map.empty
            | IdxValue idx -> Map.empty
            | Argument (name, idxs) -> Map [(name, idxs), d]
        | Unary (op, a) ->
            match op with
            | Negate -> -d |> rds a
            | Abs -> d * sgn a |> rds a
            | Sgn -> Map.empty
            | Log -> d * (a ** -1.0) |> rds a
            | Log10 -> d |> rds (log a / log 10.0)
            | Exp -> d * exp a |> rds a
            | Tanh -> d * (1.0 - (tanh a)**2.0) |> rds a
            | Sqrt -> d * (1.0 / (2.0 * sqrtt a)) |> rds a
        | Binary (op, a, b) ->
            let (.+) da db =
                let aDeriv = rds a da
                let bDeriv = rds b db
                (aDeriv, bDeriv)
                ||> Map.fold (fun m v vg -> match Map.tryFind v m with
                                            | Some ovg -> m |> Map.add v (vg + ovg)
                                            | None -> m |> Map.add v vg)                 
            match op with
            | Add -> d .+ d
            | Substract -> d .+ (-d)
            | Multiply -> (d * b) .+ (a * d)
            | Divide -> d |> rds (a * b ** -1.0)
            | Modulo -> failwith "buggy"
            | Power ->  (d * b * a**(b - 1.0)) .+ (d * a**b * log a)


    let derivFunc (fn: ElemFunc) =

        // get dimension names and add constant bias dimension
        let funcIdxNames1 = fn.DimName @ ["1"]
        let funcIdxRngs1 = (fn.Shape |> List.map (fun high -> 0L, high)) @ [1L, 1L]

        // incoming derivative w.r.t. function
        let dExprArgName = sprintf "d%s" fn.Name
        let dExpr = arg dExprArgName (fn.DimName |> List.map (fun dim -> IdxExpr.factor dim Rat.One))
        let dArgShapes = fn.ArgShape |> Map.add dExprArgName fn.Shape

        let processArg argName (IdxExprs argIdxs) dArg =
            // name the indices of the argument
            let argIdxNames = argIdxs |> List.mapi (fun i _ -> sprintf "d%s_%d" argName i)
            let argIdxSizes = argIdxNames |> List.mapi (fun i name -> name, fn.ArgShape.[argName].[i]) |> Map.ofList

            // add "1" dimension to indices
            let argIdxs1 = argIdxs @ [IdxExpr.one]
            let argIdxNames1 = argIdxNames @ ["1"]

            // Construct matrix mapping from function indices to argument indices: argIdxMat[argDim, funcDim] 
            let argIdxMat = IdxExprs.toMatrix funcIdxNames1 (IdxExprs argIdxs1)

            // Compute inverse of it.
            let ci = Consumers.compute (Tensor.convert<bigint> argIdxMat) funcIdxRngs1

            // For now assume 1:1 mapping.
            // Get matrix mapping from argument indices to function indices: funcIdxMat[funcDim, argDim] 
            let funcIdxMat = ci.YToX |> HostTensor.toList2D

            let toIdxExpr (names: string list) (facs: Rat list) =
                List.zip names facs
                |> List.fold (fun expr (name, fac) -> expr + IdxExpr.factor name fac) IdxExpr.zero

            // Substitute function indices with argument indices in derivative.
            let subs =
                List.zip funcIdxNames1 funcIdxMat
                |> List.map (fun (name, argFacs) -> name, toIdxExpr argIdxNames1 argFacs)
                |> Map.ofList
                |> Map.add "1" IdxExpr.one
            let dArgExpr = substIdx subs dArg 

            let limitIdxExprs (rng: FourierMotzkin.Range) (substSyms: string list) =
                let lows, highs = List.unzip ci.Ranges
                let lows = lows |> HostTensor.ofList |> Tensor.convert<Rat>
                let highs = highs |> HostTensor.ofList |> Tensor.convert<Rat>            
                let bLowConst = rng.BLow .* Tensor.concat 0 [lows; -highs] |> HostTensor.toList
                let bHighConst = rng.BHigh .* Tensor.concat 0 [lows; -highs] |> HostTensor.toList
                let bLowMat = rng.BLow .* Tensor.concat 0 [-ci.YToX; ci.YToX] |> HostTensor.toList2D
                let bHighMat = rng.BHigh .* Tensor.concat 0 [-ci.YToX; ci.YToX] |> HostTensor.toList2D
                let sLowMat = rng.SLow |> HostTensor.toList2D
                let sHighMat = rng.SHigh |> HostTensor.toList2D
                // now make it an index expression                
                let lows = 
                    List.zip3 bLowConst bLowMat sLowMat
                    |> List.map (fun (c, bFacs, sFacs) -> 
                        c * IdxExpr.one + toIdxExpr argIdxNames1 bFacs + toIdxExpr substSyms sFacs)
                let highs = 
                    List.zip3 bHighConst bHighMat sHighMat
                    |> List.map (fun (c, bFacs, sFacs) -> 
                        c * IdxExpr.one + toIdxExpr argIdxNames1 bFacs + toIdxExpr substSyms sFacs)
                lows, highs

            let rec buildSum summand sols sumSyms =
                match sols with
                | FourierMotzkin.Feasibility fs :: rSols ->
                    // TODO
                    buildSum summand rSols sumSyms
                | FourierMotzkin.Range rng :: rSols ->
                    let lows, highs = limitIdxExprs rng sumSyms
                    let sumSym = sprintf "d%s_z%d" argName rng.Idx
                    let summand = buildSum summand rSols (sumSym::sumSyms)
                    sum sumSym lows highs summand
                | [] -> summand

            let dArgExpr = buildSum dArgExpr ci.ConstraintsLeft []

            //printfn "arg: %s  nullspace:%A" argName ci.Nullspace

            // build function
            func (sprintf "d%s" argName) argIdxNames argIdxSizes dArgShapes dArgExpr

        // calculate derivative expressions w.r.t. all arguments
        let dArgIdxExprs = derivExpr fn.Expr dExpr
            
        // perform index substitution on the derivatives of all arguments
        let dArgIdxFns =
            dArgIdxExprs
            |> Map.toList
            |> List.map (fun ((argName, argIdxs), dArg) -> processArg argName argIdxs dArg)

        // sum by argument
        let dArgFns = 
            dArgIdxFns
            |> List.groupBy (fun ef -> ef.Name)
            |> List.map (fun (dArgName, dArgs) -> 
                dArgName, dArgs |> List.reduce (fun a {Expr=bExpr} -> {a with Expr=a.Expr + bExpr}))
            |> Map.ofList

        dArgFns



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
