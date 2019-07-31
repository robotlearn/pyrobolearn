/**
 * Type converters used to convert between different data types.
 *
 * Copyright (c) 2019, Brian Delhaisse <briandelhaisse@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "converter.hpp"

namespace py = pybind11;


/// \brief: convert from raisim::VecDyn to np.array[float64[n]]
py::array_t<double> convert_vecdyn_to_np(const raisim::VecDyn &vec) {
    const double *ptr = vec.ptr();  // get data pointer
    size_t n = vec.n; // get dimension

    // return np.array[float64[n,m]]
    return py::array_t<double>(
        {n},     // shape
        {sizeof(double)},   // C-style contiguous strides for double (double=8bytes)
        ptr);
//        vec);   // numpy array references this parent
}


/// \brief: convert from np.array[float[n]] to raisim::VecDyn
raisim::VecDyn convert_np_to_vecdyn(py::array_t<double> array) {

    size_t size = array.size();

    // reshape if necessary
    if (array.ndim() > 1)
        array.resize({size});

    // create raisim dynamic vector
    raisim::VecDyn vec(size);

    // copy the data
    for(size_t i=0; i<size; i++) {
        vec[i] = *array.data(i);
    }

    // return vector
    return vec;
}


/// \brief: convert from raisim::MatDyn to np.array[float64[n,m]]
py::array_t<double> convert_matdyn_to_np(const raisim::MatDyn &mat) {
    const double *ptr = mat.ptr();  // get data pointer
    size_t n = mat.n;
    size_t m = mat.m;

    // return np.array[float64[n,m]]
    return py::array_t<double>(
        {n, m},     // shape
        {sizeof(double), sizeof(double)},   // C-style contiguous strides for double (double=8bytes)
        ptr);
//        mat);   // numpy array references this parent
}


/// \brief: convert from np.array[float[n,m]] to raisim::MatDyn
raisim::MatDyn convert_np_to_matdyn(py::array_t<double> array) {

    // check dimensions and shape
    if (array.ndim() != 2) {
        std::ostringstream s;
        s << "error: expecting the given array to have a dimension of 2, but got instead a dimension of "
            << array.ndim() << ".";
        throw std::domain_error(s.str());
    }

    // get the number of rows and columns
    size_t nrows = array.shape(0);
    size_t ncols = array.shape(1);

    // create raisim matrix
    raisim::MatDyn mat(nrows, ncols);

    // copy the data
    for (size_t i=0; i<nrows; i++)
        for (size_t j=0; j<ncols; j++)
            mat[i, j] = *array.data(i, j);

    // return matrix
    return mat;
}
