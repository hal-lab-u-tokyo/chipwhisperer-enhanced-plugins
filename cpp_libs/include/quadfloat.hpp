#ifndef QUADFLOAT_H
#define QUADFLOAT_H

namespace QuadFloat {
	typedef struct {
		double hi;
		double lo;
	} double2;

	class QF128 {
	private:
		double hi, lo;

	public:
		QF128() : hi(0), lo(0) {}
		QF128(double x) : hi(x), lo(0) {}
		QF128(double hi, double lo) : hi(hi), lo(lo) {}
		QF128(int x) : hi((double)x), lo(0) {}
		QF128(long int x) : hi((double)x), lo(0) {}
		QF128(long long x) : hi((double)x), lo(0) {}
		// cast to double
		explicit operator double() const {
			return hi + lo;
		}

		QF128 operator+(QF128 x) const;
		QF128 operator-(QF128 x) const {
			return *this + (-x);
		}
		QF128 operator*(QF128 x) const;

		QF128 operator-() const {
			return QF128(-hi, -lo);
		}
		QF128 operator+=(const QF128 &x) {
			*this = *this + x;
			return *this;
		}
		QF128 operator-=(const QF128 &x) {
			*this = *this - x;
			return *this;
		}
		QF128 operator*=(const QF128 &x) {
			*this = *this * x;
			return *this;
		}
		QF128 operator+(const double &x) const {
			return *this + QF128(x);
		}
		QF128 operator-(const double &x) const {
			return *this - QF128(x);
		}
		QF128 operator*(const double &x) const {
			return *this * QF128(x);
		}

		QF128 operator+(const int &x) const {
			return *this + QF128(x);
		}

		QF128 operator-(const int &x) const {
			return *this - QF128(x);
		}

		QF128 operator*(const int &x) const {
			return *this * QF128(x);
		}

		QF128 operator+(const long int &x) const {
			return *this + QF128(x);
		}

		QF128 operator-(const long int &x) const {
			return *this - QF128(x);
		}

		QF128 operator*(const long int &x) const {
			return *this * QF128(x);
		}


		QF128 operator+(const long long &x) const {
			return *this + QF128(x);
		}

		QF128 operator-(const long long &x) const {
			return *this - QF128(x);
		}

		QF128 operator*(const long long &x) const {
			return *this * QF128(x);
		}
	};
};

#endif //QUADFLOAT_H