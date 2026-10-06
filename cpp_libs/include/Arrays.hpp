/*
*    Copyright (C) 2024 The University of Tokyo
*    
*    File:          /cpp_libs/include/Arrays.hpp
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  23-01-2024 16:56:46
*    Last Modified: 01-05-2025 08:02:04
*/


#ifndef ARRAYS_H
#define ARRAYS_H

#include <tuple>

template <typename T>
class Array2D {
public:
	// Constructors
	Array2D(T* data, size_t dim_x, size_t dim_y) : data(data), dim_x(dim_x), dim_y(dim_y), need_free(false) {};

	Array2D(T* data, std::tuple<size_t, size_t> dim) : Array2D(data, std::get<0>(dim), std::get<1>(dim)) {};

	Array2D(size_t dim_x, size_t dim_y) : dim_x(dim_x), dim_y(dim_y), need_free(true), data(new T[dim_x * dim_y]()) {};

	Array2D(std::tuple<size_t, size_t> dim) : Array2D(std::get<0>(dim), std::get<1>(dim)) {};

	// Destructor
	~Array2D() {
		if (need_free) {
			delete[] data;
		}
	}

	T &at(size_t x, size_t y) {
		return data[x * dim_y + y];
	}

	const T* get_pointer() {
		return data;
	}

	std::tuple<size_t, size_t> get_dim() {
		return std::make_tuple(dim_x, dim_y);
	}

	size_t get_size() {
		return dim_x * dim_y * sizeof(T);
	}

private:
	size_t dim_x, dim_y;
	bool need_free = false;
	T* const data;
};

template <typename T>
class Array3D {
public:

	// Constructors
	Array3D(T* data, size_t dim_x, size_t dim_y, size_t dim_z) : data(data), dim_x(dim_x), dim_y(dim_y), dim_z(dim_z), need_free(false) {};

	Array3D(T* data, std::tuple<size_t, size_t, size_t> dim) : Array3D(data, std::get<0>(dim), std::get<1>(dim), std::get<2>(dim)) {};

	Array3D(size_t dim_x, size_t dim_y, size_t dim_z) : dim_x(dim_x), dim_y(dim_y), dim_z(dim_z), need_free(true), data(new T[dim_x * dim_y * dim_z]()) {};

	Array3D(std::tuple<size_t, size_t, size_t> dim) : Array3D(std::get<0>(dim), std::get<1>(dim), std::get<2>(dim)) {};

	// Destructor
	~Array3D() {
		if (need_free) {
			delete[] data;
		}
	}

	T &at(size_t x, size_t y, size_t z) {
		return data[x * dim_y * dim_z + y * dim_z + z];
	}

	const T* get_pointer() {
		return data;
	}

	size_t get_size() {
		return dim_x * dim_y * dim_z * sizeof(T);
	}

private:
	size_t dim_x, dim_y, dim_z;
	bool need_free;
	T* const data;
};


#endif //ARRAYS_H