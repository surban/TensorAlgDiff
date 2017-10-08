
open System
open System.Numerics
open PLplot
open Tensor
open Tensor.Algorithms

open OldTests
open ConsumerTests


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


