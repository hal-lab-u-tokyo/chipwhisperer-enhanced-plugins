/*
*    Copyright (C) 2024 The University of Tokyo
*    
*    File:          /cpp_libs/include/Arrays.hpp
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  23-01-2024 16:56:46
*    Last Modified: 01-02-2025 05:17:20
*/


#ifndef ARRAYS_H
#define ARRAYS_H

#include <tuple>

template <typename T>
class Array2D {
public:
	// Constructors
	Array2D(T* data, int dim_x, int dim_y) : data(data), dim_x(dim_x), dim_y(dim_y), need_free(false) {};

	Array2D(T* data, std::tuple<int, int> dim) : Array2D(data, std::get<0>(dim), std::get<1>(dim)) {};

	Array2D(int dim_x, int dim_y) : dim_x(dim_x), dim_y(dim_y), need_free(true), data(new T[dim_x * dim_y]()) {};

	Array2D(std::tuple<int, int> dim) : Array2D(std::get<0>(dim), std::get<1>(dim)) {};

	// Destructor
	~Array2D() {
		if (need_free) {
			delete[] data;
		}
	}

	T &at(int x, int y) {
		return data[x * dim_y + y];
	}

	const T* get_pointer() {
		return data;
	}

	std::tuple<int, int> get_dim() {
		return std::make_tuple(dim_x, dim_y);
	}

	size_t get_size() {
		return dim_x * dim_y * sizeof(T);
	}

private:
	int dim_x, dim_y;
	bool need_free = false;
	T* const data;
};

template <typename T>
class Array3D {
public:

	// Constructors
	Array3D(T* data, int dim_x, int dim_y, int dim_z) : data(data), dim_x(dim_x), dim_y(dim_y), dim_z(dim_z), need_free(false) {};

	Array3D(T* data, std::tuple<int, int, int> dim) : Array3D(data, std::get<0>(dim), std::get<1>(dim), std::get<2>(dim)) {};

	Array3D(int dim_x, int dim_y, int dim_z) : dim_x(dim_x), dim_y(dim_y), dim_z(dim_z), need_free(true), data(new T[dim_x * dim_y * dim_z]()) {};

	Array3D(std::tuple<int, int, int> dim) : Array3D(std::get<0>(dim), std::get<1>(dim), std::get<2>(dim)) {};

	// Destructor
	~Array3D() {
		if (need_free) {
			delete[] data;
		}
	}

	T &at(int x, int y, int z) {
		return data[x * dim_y * dim_z + y * dim_z + z];
	}

	const T* get_pointer() {
		return data;
	}

	size_t get_size() {
		return dim_x * dim_y * dim_z * sizeof(T);
	}

private:
	int dim_x, dim_y, dim_z;
	bool need_free;
	T* const data;
};

template <typename T>
class Array4D {
public:

	// Constructors
	Array4D(T* data, int dim_w, int dim_x, int dim_y, int dim_z) : data(data), dim_w(dim_w), dim_x(dim_x), dim_y(dim_y), dim_z(dim_z), need_free(false) {};

	Array4D(T* data, std::tuple<int, int, int, int> dim) : Array4D(data, std::get<0>(dim), std::get<1>(dim), std::get<2>(dim), std::get<3>(dim)) {};

	Array4D(int dim_w, int dim_x, int dim_y, int dim_z) : dim_w(dim_w), dim_x(dim_x), dim_y(dim_y), dim_z(dim_z), need_free(true), data(new T[dim_w * dim_x * dim_y * dim_z]()) {};

	Array4D(std::tuple<int, int, int, int> dim) : Array4D(std::get<0>(dim), std::get<1>(dim), std::get<2>(dim), std::get<3>(dim)) {};


	// Destructor
	~Array4D() {
		if (need_free) {
			delete[] data;
		}
	}

	T &at(int w, int x, int y, int z) {
		return data[w * dim_x * dim_y * dim_z + x * dim_y * dim_z + y * dim_z + z];
	}

	const T* get_pointer() {
		return data;
	}

	size_t get_size() {
		return dim_w * dim_x * dim_y * dim_z * sizeof(T);
	}

private:
	int dim_w, dim_x, dim_y, dim_z;
	bool need_free;
	T* const data;
};

#endif //ARRAYS_H