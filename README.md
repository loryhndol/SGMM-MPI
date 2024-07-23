# Value-based Spatial Distribution

A header-only MPI-based C++ implementation of the paper "Statistical Visualization and Analysis of Large Data Using a Value-based Spatial Distribution"

Current support data type: float32

## API Reference

### build SGMMs

```cpp
template<typename T>
Representation<T> DataModeling(std::array<int, 3> dims, int blockSize, std::string filename, Endian endian);
```

### export data

```cpp
template<typename T>
void ExportReconstructedVolume(std::string outputFilename, Representation<T>& dataBlocks, std::array<int, 3> dims, int blockSize);

template<typename T>
void SaveParameters(std::string filename, Representation<T>& data);
```

## Binary Format of SGMM Parameters

- blockId[0], blockId[1], blockId[2]

- blockDims[0], blockDims[1], blockDims[2]

- numOfBins

- minValue

- maxValue

- histogram
  
  - binId
  
  - probability
  
  - numOfGaussians
  
  - GaussiansList
    - weight: `std::vector<T>`
    - means: `std::vector<std::array<T, 3>>`
    - covariance(upper triangular): `std::vector<T>`

## Paper Reference

Wang, Ko-Chih & Lu, Kewei & Wei, Tzu-Hsuan & Shareef, Naeem & Shen, Han-Wei. (2017). Statistical visualization and analysis of large data using a value-based spatial distribution. 161-170. 10.1109/PACIFICVIS.2017.8031590.
