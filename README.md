# Load-balance Level Coarsening (LBC)
Load-balance Level Coarsening is a scheduling algorithm for 
making sparse matrix loops parallel. It can be used within 
 code generators or libraries. For more information see 
[Sympiler website](http:://www.sympiler.com/).

## Install

### Linux
LBC library does not have any dependency.
However, if you want to run the triangular solve example, 
you need to install METIS. If METIS is installed in the system path,
CMAKE will resolve the dpendency otherwise you need to set 
`CMAKE_PREFIX_PATH` to the root directory of metis, i.e., 
where the cmakelists file exists. 
For installing METIS in Ubuntu you can also use
```
sudo apt install metis
```


Then install LBC by following commands:

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

### Mac
Setting the C and CXX compilers to GCC and then follow the Linux 
instructions. For example:
`-DCMAKE_CXX_COMPILER=/usr/local/Cellar/gcc\@9/9.3.0_2/bin/g++-9 -DCMAKE_C_COMPILER=/usr/local/Cellar/gcc\@9/9.3.0_2/bin/gcc-9`

Alternatively, you can use CLang using `brew install llvm`. 
The default clang on Mac might not work so make sure to set it llvm clang:
`-DCMAKE_C_COMPILER=/usr/local/opt/llvm/bin/clang -DCMAKE_CXX_COMPILER=/usr/local/opt/llvm/bin/clang++`

## Example
As an example, sparse triangular solver example, CSR is turned into
a parallel code using the LBC algorithm. 


