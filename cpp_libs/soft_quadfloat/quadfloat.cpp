#include <quadfloat.hpp>
#include <cmath>

using namespace QuadFloat;

inline double2 quickTwoSum(double a, double b) {
	double s = a + b;
	double e = b - (s - a);
	return {s, e};
}

inline double2 twoSum(double a, double b) {
	double s = a + b;
	double v = s - a;
	double e = (a - (s - v)) + (b - v);
	return {s, e};
}

inline double2 twoProdFMA(double a, double b) {
	double p = a * b;
	double e = std::fma(a, b, -p);
	return {p, e};
}

QF128 QF128::operator+(QF128 x) const
{
	double2 s = twoSum(hi, x.hi);
	double2 t = twoSum(lo, x.lo);
	s.lo += t.hi;
	s = quickTwoSum(s.hi, s.lo);
	s.lo += t.lo;
	auto c = quickTwoSum(s.hi, s.lo);
	return QF128(c.hi, c.lo);
}

QF128 QF128::operator*(QF128 x) const
{
	double2 p = twoProdFMA(hi, x.hi);
	p.hi += hi * x.lo;
	p.hi += lo * x.hi;
	p = quickTwoSum(p.hi, p.lo);
	return QF128(p.hi, p.lo);
}
