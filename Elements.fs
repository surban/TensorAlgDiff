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
            static member constVal (IdxExpr f) =
                match f |> Map.tryFind "1" with
                | Some v -> v
                | None -> Rat.Zero
            static member ofSeq indices values =
                Seq.zip indices values
                |> Map.ofSeq
                |> IdxExpr

    /// Matches an index expression that consists only of a constant.
    let (|ConstIdxExpr|_|) (IdxExpr f) =
        let f = f |> Map.toList |> List.filter (fun (_, v) -> v <> Rat.Zero)
        match f with
        | [] -> Some Rat.Zero
        | [i, v] when i = "1" -> Some v
        | _ -> None

    /// Matches an index expression that consists only of a single (non-constant) factor.
    let (|SingleIdxExpr|_|) (IdxExpr f) =
        let f = f |> Map.toList |> List.filter (fun (_, v) -> v <> Rat.Zero)
        match f with
        | [i, v] when i <> "1" -> Some (i, v)
        | _ -> None
   

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

    and BinaryOp = 
        | Add                           
        | Substract                     
        | Multiply                      
        | Divide                        
        | Modulo
        | Power        
        | IdxIf of idx:IdxExpr * cmp:IdxComparison

    and IdxComparison =
        | EqualToZero
        | GreaterOrEqualToZero

    /// an element expression
    and [<StructuredFormatDisplay("{Pretty}")>]
        ElemExpr =
        | Leaf of LeafOp
        | Unary of UnaryOp * ElemExpr
        | Binary of BinaryOp * ElemExpr * ElemExpr

    and [<StructuredFormatDisplay("{Pretty}")>]
        ElemFunc = {
            Name:       string
            DimNames:    string list
            DimSize:    Map<string, int64>
            Expr:       ElemExpr
            ArgShapes:   Map<string, int64 list>
        } with
            member this.Pretty =
                let dims = this.DimNames |> String.concat "; "
                sprintf "%s[%s] = %A" this.Name dims this.Expr
            member this.Shape = 
                this.DimNames |> List.map (fun d -> this.DimSize.[d])

    /// Returns all arguments occuring in the given expression/
    let rec extractArgs expr =
        match expr with
        | Leaf (Argument (name, idxs)) -> Set [name, idxs]
        | Leaf _ -> Set.empty
        | Unary (_, a) -> extractArgs a
        | Binary (_, a, b) -> Set.union (extractArgs a) (extractArgs b)

    /// Builds a function.
    let func name dimNames dimSizes argShapes expr =
        for (argName, argIdx) in extractArgs expr do
            match argShapes |> Map.tryFind argName with
            | Some shp when IdxExprs.length argIdx <> List.length shp -> 
                failwithf "shape dimensionality mismatch for argument %s" argName
            | Some shp -> ()
            | None -> failwithf "no shape specified for argument %s" argName                
        {Name=name; DimNames=dimNames; DimSize=dimSizes; Expr=expr; ArgShapes=argShapes}

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
                    | Sum (sym, lows, highs) -> 
                        let lowsStr =
                            match lows with
                            | [ConstIdxExpr low] -> sprintf "%A" low
                            | [low] -> sprintf "(%A)" low
                            | _ -> sprintf "(max %A)" lows
                        let highsStr =
                            match highs with
                            | [ConstIdxExpr high] -> sprintf "%A" high
                            | [high] -> sprintf "(%A)" high
                            | _ -> sprintf "(min %A)" highs
                        sprintf "sum{%s}_%s^%s (%s)" sym lowsStr highsStr aStr
                myStr, myPri
                
            | Binary(op, a, b) -> 
                let aStr, aPri = a.PrettyAndPriority
                let bStr, bPri = b.PrettyAndPriority            
                match op with
                | Add | Substract | Multiply | Divide | Modulo | Power ->
                    let mySym, myPri =
                        match op with
                        | Add -> "+", 1
                        | Substract -> "-", 1
                        | Multiply -> "*", 2
                        | Divide -> "/", 2
                        | Modulo -> "%", 2
                        | Power -> "**", 5
                        | _ -> failwith "unexpected"
                    let aStr =
                        if myPri > aPri then sprintf "(%s)" aStr
                        else aStr
                    let bStr =
                        if myPri > bPri then sprintf "(%s)" bStr
                        else bStr
                    let myStr = sprintf "%s %s %s" aStr mySym bStr
                    myStr, myPri            
                | IdxIf (idx, cmp) ->
                    let cmpStr =
                        match cmp with
                        | GreaterOrEqualToZero -> ">= 0"
                        | EqualToZero -> "= 0"
                    sprintf "if {%A %s} then (%s) else (%s)" idx cmpStr aStr bStr, 0

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

    /// Summation over an index.
    let sum idx lows highs a =
        Unary (Sum (idx, lows, highs), a)

    /// Expression conditioned on index values.
    let idxIf idx cmp thenExpr elseExpr =
        match cmp, idx with
        | EqualToZero, ConstIdxExpr v when v = Rat.Zero -> thenExpr
        | EqualToZero, ConstIdxExpr v -> elseExpr
        | GreaterOrEqualToZero, ConstIdxExpr v when v >= Rat.Zero -> thenExpr
        | GreaterOrEqualToZero, ConstIdxExpr v -> elseExpr
        | _ -> Binary (IdxIf (idx, cmp), thenExpr, elseExpr)

    /// Substitutes the specified size symbols with their replacements.
    let rec substIdx repl expr = 
        let sub = substIdx repl
        match expr with
        | Leaf (IdxValue idx) -> Leaf (IdxValue (IdxExpr.subst repl idx))
        | Leaf (Argument (name, idxs)) -> Leaf (Argument (name, IdxExprs.subst repl idxs))
        | Leaf (op) -> Leaf (op)
        | Unary (Sum (idx, lows, highs), a) ->
            Unary (Sum (idx, lows |> List.map (IdxExpr.subst repl), highs |> List.map (IdxExpr.subst repl)), sub a)
        | Unary (op, a) -> Unary (op, sub a)
        | Binary (IdxIf (idx, cmp), a, b) -> 
            Binary (IdxIf (idx |> IdxExpr.subst repl, cmp), sub a, sub b)
        | Binary (op, a, b) -> Binary (op, sub a, sub b)

    /// Evaluates the given expression.
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
            match op with
            | Negate -> -(subEval a) 
            | Abs -> abs (subEval a) 
            | Sgn -> Operators.sgn (subEval a)
            | Log -> log (subEval a)
            | Log10 -> log10 (subEval a)
            | Exp -> exp (subEval a)
            | Tanh -> tanh (subEval a)
            | Sqrt -> sqrt (subEval a)
            | Sum (sym, lows, highs) ->
                let low = lows |> List.map (IdxExpr.eval idxEnv) |> List.max |> ceil 
                let high = highs |> List.map (IdxExpr.eval idxEnv) |> List.min |> floor
                seq {low .. high}
                |> Seq.map (fun v -> evalExpr argEnv (idxEnv |> Map.add sym v) a)
                |> Seq.sum

        | Binary (op, a, b) ->
            match op with
            | Add -> (subEval a) + (subEval b)
            | Substract -> (subEval a) - (subEval b)                
            | Multiply -> (subEval a) * (subEval b)          
            | Divide -> (subEval a) / (subEval b)           
            | Modulo -> (subEval a) % (subEval b)
            | Power -> (subEval a) ** (subEval b)
            | IdxIf (idx, cmp) ->
                let idxVal = idx |> IdxExpr.eval idxEnv
                match cmp with
                | EqualToZero when idxVal = Rat.Zero -> subEval a
                | EqualToZero -> subEval b
                | GreaterOrEqualToZero when idxVal >= Rat.Zero -> subEval a
                | GreaterOrEqualToZero -> subEval b

    /// Evaluates the given function.
    let evalFunc argEnv (func: ElemFunc) =
        let fv = HostTensor.zeros func.Shape
        for pos in TensorLayout.allIdxOfShape func.Shape do
            let idxEnv =
                List.zip pos func.DimNames
                |> List.fold (fun env (p, name) -> env |> Map.add name (Rat p)) Map.empty
            fv.[pos] <- evalExpr argEnv idxEnv func.Expr
        fv

    /// Calculates the derivative expression given the incoming derivative dExpr.
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
            | Sum _ -> failwith "sum derivative not implemented yet"
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
            | IdxIf (idx, cmp) ->
                (idxIf idx cmp d (scalar 0.0)) .+ (idxIf idx cmp (scalar 0.0) d)


    /// Calculates the derivative functions of y w.r.t. all of its arguments.
    let derivFunc (y: ElemFunc) =
        // get dimension names and add constant bias dimension
        let yIdxNames1 = y.DimNames @ ["1"]
        let yIdxRngs1 = (y.Shape |> List.map (fun size -> 0L, size-1L)) @ [1L, 1L]
        let yLows1, yHighs1 = List.unzip yIdxRngs1
        let yLows1 = yLows1 |> HostTensor.ofList |> Tensor.convert<Rat>
        let yHighs1 = yHighs1 |> HostTensor.ofList |> Tensor.convert<Rat>            

        // incoming derivative dy w.r.t. function y
        let dyArgName = sprintf "d%s" y.Name
        let dy = arg dyArgName (y.DimNames |> List.map (fun d -> IdxExpr.factor d Rat.One))
        let argShapes = y.ArgShapes |> Map.add dyArgName y.Shape

        // Calculate derivative expressions w.r.t. all indiced arguments.
        let dxs = derivExpr y.Expr dy            

        // Perform index substitution and nullspace summation on the derivatives of all arguments.
        let processDeriv xName (IdxExprs xIdxs) dx =
            // name the argument and its indices
            let dxName = sprintf "d%s" xName
            let dxIdxNames = xIdxs |> List.mapi (fun i _ -> sprintf "%s_%d" dxName i)
            let dxIdxSizes = dxIdxNames |> List.mapi (fun i name -> name, y.ArgShapes.[xName].[i]) |> Map.ofList

            // Add "1" dimension to indices for constant terms.
            let dxIdxs1, dxIdxNames1 = xIdxs @ [IdxExpr.one], dxIdxNames @ ["1"]

            // Construct matrix mapping from function indices to argument indices yToX[xDim, yDim] and compute
            // the generalized inverse of it.
            let yToX = IdxExprs.toMatrix yIdxNames1 (IdxExprs dxIdxs1) |> Tensor.convert<bigint>
            let inv = Consumers.compute yToX yIdxRngs1
                              
            // Perform summation over nullspace.
            let rec buildSum summand sols sumSyms =
                match sols with
                | FourierMotzkin.Feasibility fs :: rSols ->
                    let summand = buildSum summand rSols sumSyms
                    // System is feasible if fs .* b <= 0,
                    // where b = [yLows - YToX .* y; -yHighs + YToX .* y].
                    let bVec = fs .* Tensor.concat 0 [yLows1; -yHighs1] |> HostTensor.toList
                    let bMat = fs .* Tensor.concat 0 [-inv.YToX; inv.YToX] |> HostTensor.toList2D
                    let fsIdxs = 
                        List.zip bVec bMat
                        |> List.map (fun (c, bFacs) -> c * IdxExpr.one + IdxExpr.ofSeq dxIdxNames1 bFacs)
                        |> List.filter (fun ie ->
                            // Filter inequalaties that are always true.
                            // Each inequality of the form cv + iv * "i" <= 0 is considered.
                            let cv = IdxExpr.constVal ie
                            match ie - cv * IdxExpr.one with 
                            // cv - "i" <= 0 => cv <= "i" => always true for cv <= 0 because "i" >= 0                                         
                            | SingleIdxExpr (i, iv) when iv = Rat.MinusOne && cv <= Rat.Zero -> false
                            // cv + "i" <= 0 => "i" <= -cv => always true for -cv >= size_i-1 because "i" <= size_i-1
                            | SingleIdxExpr (i, iv) when iv = Rat.One && -cv >= Rat (dxIdxSizes.[i]-1L) -> false
                            | _ -> true)         
                    (summand, fsIdxs) ||> List.fold (fun s fsIdx -> idxIf -fsIdx GreaterOrEqualToZero s (scalar 0.0))                    
                | FourierMotzkin.Range rng :: rSols ->
                    let sumSym = sprintf "%s_z%d" dxName rng.Idx
                    let summand = buildSum summand rSols (sumSym::sumSyms)
                    // The limits are given by 
                    // Low limits:  x[Idx] >= BLow  .* b - SLow  .* x.[Idx+1L..]
                    // High limits: x[Idx] <= BHigh .* b - SHigh .* x.[Idx+1L..]
                    // where b = [yLows - YToX .* y; -yHighs + YToX .* y].              
                    let bVec = Tensor.concat 0 [yLows1; -yHighs1]
                    let bMat = Tensor.concat 0 [-inv.YToX; inv.YToX]
                    let bLowVec = rng.BLow .* bVec |> HostTensor.toList
                    let bHighVec = rng.BHigh .* bVec |> HostTensor.toList
                    let bLowMat = rng.BLow .* bMat |> HostTensor.toList2D
                    let bHighMat = rng.BHigh .* bMat |> HostTensor.toList2D
                    let sLowMat = rng.SLow |> HostTensor.toList2D
                    let sHighMat = rng.SHigh |> HostTensor.toList2D
                    let idxExpr bVec bMat sMat = 
                        List.zip3 bVec bMat sMat
                        |> List.map (fun (c, bFacs, sFacs) -> 
                            c * IdxExpr.one + IdxExpr.ofSeq dxIdxNames1 bFacs + IdxExpr.ofSeq sumSyms sFacs)
                    let lows, highs = idxExpr bLowVec bLowMat sLowMat, idxExpr bHighVec bHighMat sHighMat
                    sum sumSym lows highs summand
                | [] -> 
                    let yToX = inv.YToX |> HostTensor.toList2D
                    let zToX = inv.Nullspace |> Tensor.convert<Rat> |> HostTensor.toList2D
                    let subs =
                        List.zip3 yIdxNames1 yToX zToX
                        |> List.map (fun (name, argFacs, nsFacs) -> 
                            name, IdxExpr.ofSeq dxIdxNames1 argFacs + IdxExpr.ofSeq sumSyms nsFacs)
                        |> Map.ofList
                        |> Map.add "1" IdxExpr.one
                    substIdx subs summand 
            let dxSummed = buildSum dx inv.ConstraintsLeft []

            // Check solvability.
            let solIdxs = 
                inv.Solvability 
                |> Tensor.convert<Rat> 
                |> HostTensor.toList2D
                |> List.map (fun sFacs -> IdxExpr.ofSeq dxIdxNames1 sFacs)
            let dxChecked = 
                (dxSummed, solIdxs) ||> List.fold (fun s solIdx -> idxIf solIdx EqualToZero s (scalar 0.0))

            // Build derivative function.
            func dxName dxIdxNames dxIdxSizes argShapes dxChecked

        // Perform index substitution on the derivatives of all arguments and sum by argument.
        let dxFns = 
            dxs 
            |> Map.toList 
            |> List.map (fun ((xName, xIdxs), dx) -> processDeriv xName xIdxs dx)
            |> List.groupBy (fun ef -> ef.Name)
            |> List.map (fun (dxName, dxs) -> 
                dxName, dxs |> List.reduce (fun a {Expr=bExpr} -> {a with Expr=a.Expr + bExpr}))
            |> Map.ofList

        dxFns

