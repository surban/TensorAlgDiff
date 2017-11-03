Automatic Differentiation for Tensor Algebras
=============================================

Read the corresponding [technical report for more details and examples](https://github.com/surban/TensorAlgDiff/raw/master/elemdiff.pdf).

This code computes expressions for the derivatives of element-wise defined tensor-valued functions.
It can handle arguments inside the functions that are indexed by arbitrary linear combinations of the function indices.
Furthermore, the function may contain (nested) sums with arbitrary ranges (even linearly depending on other indices).
An example is the matrix-valued function f(a,b,c,d), which is element-wise defined by the expression

    f[i; j] = exp (-sum{k}_0^4 (((a[i; k] + b[j; k]) ** 2 * c[i; i] + d[i + k] ** 3)))

Deriving derivative expressions when the mapping between function indices and argument indices is not 1:1 requires special attention.
For example, for the function `f_{ij} (x) = x_i^2`, the derivative of some loss `l=l(f(x))` w.r.t. `x` is `(dx)_i = dl / dx_i = \sum_j (df)_{ij} 2 x_i`; the sum is necessary because index `j` does not appear in the indices of `f`.
Another example is `f_i (x) = x_{ii}^2`, where `x` is a matrix; here we have `(dx)_{ij} = \kronecker_{i=j} (df)_i 2 x_{ii}`; the Kronecker delta is necessary because the derivative is zero for off-diagonal elements.
Another indexing scheme is used by `f_{ij} (x) = exp x_{i+j}`; here the correct derivative is `(dx)_k = \sum_i (df)_{i,k-i} \exp x_k`, where the range of the sum must be chosen appropriately.

Our algorithm can handle any case in which the indices of an argument are an *arbitrary linear combination* of the indices of the function, thus all of the above examples can be handled.
Sums (and their ranges) and Kronecker deltas are automatically inserted into the derivatives as necessary.
Additionally, the indices are transformed, if required (as in the last example).
The algorithm outputs a symbolic expression that can be subsequently fed into a tensor algebra compiler.

For the above expression the algorithm outputs:

    Derivative of f wrt. a: da[da_0; da_1] = sum{da_z0}_0^3 (((-(df[da_0; da_z0] * exp (-sum{k}_0^4 (((a[da_0; k] + b[da_z0; k]) ** 2 * c[da_0; da_0] + d[da_0 + k] ** 3))))) * c[da_0; da_0] * 2 * (a[da_0; da_1] + b[da_z0; da_1]) ** (2 - 1)))
    Derivative of f wrt. b: db[db_0; db_1] = sum{db_z0}_0^2 (((-(df[db_z0; db_0] * exp (-sum{k}_0^4 (((a[db_z0; k] + b[db_0; k]) ** 2 * c[db_z0; db_z0] + d[db_z0 + k] ** 3))))) * c[db_z0; db_z0] * 2 * (a[db_z0; db_1] + b[db_0; db_1]) ** (2 - 1)))
    Derivative of f wrt. c: dc[dc_0; dc_1] = if {dc_0 + -dc_1 = 0} then (sum{dc_z1}_0^4 (sum{dc_z0}_0^3 (((a[dc_1; dc_z1] + b[dc_z0; dc_z1]) ** 2 * (-(df[dc_1; dc_z0] * exp (-sum{k}_0^4 (((a[dc_1; k] + b[dc_z0; k]) ** 2 * c[dc_1; dc_1] + d[dc_1 + k] ** 3))))))))) else (0)
    Derivative of f wrt. d: dd[dd_0] = sum{dd_z1}_(max [0; -2 + dd_0])^(min [4; dd_0]) (sum{dd_z0}_0^3 (((-(df[dd_0 + -dd_z1; dd_z0] * exp (-sum{k}_0^4 (((a[dd_0 + -dd_z1; k] + b[dd_z0; k]) ** 2 * c[dd_0 + -dd_z1; dd_0 + -dd_z1] + d[dd_0 + -dd_z1 + k] ** 3))))) * 3 * d[dd_0] ** (3 - 1))))

Internally, the derivatives are stored as computational trees to avoid repeated computations and thus expression blowup that otherwise occurs in symbolic differentiation.
This work can easily be employed in system that generate C++ or CUDA code for expressions or be combined with a Tensor Algebra Compiler like http://tensor-compiler.org.

Running
-------
1. Install .NET Core 2.0 from https://www.microsoft.com/net/learn/get-started/ (packages available for all operating systems).
We tested our code on Ubuntu Linux.

2. Add the feed for our Tensor library to your NuGet configuration. 
This can be done by adding the following line to the file `~/.nuget/NuGet.Config`
        
        <add key="CorePorts" value="https://www.myget.org/F/coreports/api/v3/index.json" protocolVersion="3" /> 

3. To build and run the demo execute `dotnet run`

4. To run the numeric verification tests run `dotnet test` (takes approx. 2 minutes)

Reference
---------
When using this work or the provided code please refer to the following publication.
   
    Sebastian Urban, Patrick van der Smagt. Automatic Differentiation for Tensor Algebras. arXiv cs.SC, Nov 2017.

Note that we employ some algorithms implemented in our open-source Tensor library; their source is at https://github.com/DeepMLNet/DeepNet/blob/core2/Numeric/Tensor/LinAlg.fs.


License
-------
Apache License 2.0

