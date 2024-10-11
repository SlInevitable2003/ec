#include <bits/stdc++.h>
using namespace std;

// debug helper
#define trace cout << __LINE__ << endl;
#define whatis(x) cout << __LINE__ << ": " << #x << " = " << x << endl;

// typedef of fixed (1, 32, 64) bit-length integer
typedef bool u1i;
typedef uint32_t u32i;
typedef uint64_t u64i;

// some bit-operation tools
#define LOW1(x) ((x) & u64i(1))
#define LOW16(x) ((x) & ((u64i(1) << 16) - 1))
#define LOW32(x) ((x) & ((u64i(1) << 32) - 1))
#define HIGH1(x) ((x) >> 63)
#define HIGH32(x) ((x) >> 32)
#define COMBINE(x, y) (((x) << 32) | LOW32(y))
#define OVERFLOW(x) (HIGH32(x) != 0)

// random generator

u64i urand64() { u64i res{0}; for (int i = 0; i < 4; i++) res = (res << 16) | LOW16(rand()); return res; }

// basic arithmetic module
u64i add64_with_carry(u64i x, u64i y, u1i *c_out = 0, u1i c_in = 0)
{
    u64i z{x + y + c_in};
    // z < x or z < y clearly implies an overflow,
    // z >= x and z >= y indicates no overflow except x = y = z = 2^64 - 1.
    if (c_out) *c_out = (z < x || z < y || (((~x) | (~y)) == 0));
    return z;
}
u64i sub64_with_carry(u64i x, u64i y, u1i *c_out = 0, u1i c_in = 0)
{
    u64i z{x - y - c_in};
    if (c_out) *c_out = (x < y + c_in || (c_in && ((~y) == 0)));
    return z;
}
pair<u64i, u64i> mul64(u64i x, u64i y)
{
    u64i xl{LOW32(x)}, xh{HIGH32(x)}, yl{LOW32(y)}, yh{HIGH32(y)};
    u64i p0{xl * yl}, p1{xl * yh}, p2{xh * yl}, p3{xh * yh};
    u64i u{p3}, v{p0};
    u1i carry{0};

    v = add64_with_carry(v, LOW32(p1) << 32, &carry);
    u += HIGH32(p1) + carry;
    v = add64_with_carry(v, LOW32(p2) << 32, &carry);
    u += HIGH32(p2) + carry;
    return {u, v};
}
pair<u64i, u64i> left_shift_by_one(u64i u, u64i v, u1i *c)
{   // left shift 128-bit integer by one
    *c = HIGH1(u);
    u = (u << 1) | HIGH1(v);
    v <<= 1;
    return {u, v};
}

// high-resolution integer type
template<size_t t>
class BigInt {
public:
    u64i val[t] = {0};

    BigInt(u64i sml) { val[0] = sml; }
    BigInt(u64i* arr, size_t l) { for (int i = 0; i < l; i++) val[i] = arr[i]; }

    // useful tools
    u64i operator[](size_t idx) const { return val[idx]; }
    u64i& operator[](size_t idx) { return val[idx]; }
    void print() const
    {
        cout << "0x";
        for (int i = t - 1; i >= 0; i--) cout << hex << setw(16) << setfill('0') << (*this)[i];
        cout << dec << endl;
    }
    static BigInt<t> sample() 
    {   // return a uniformly randomly selected BigInt<t> entity
        BigInt<t> res{0};
        for (int i = 0; i < t; i++) res[i] = urand64();
        return res;
    }
    
    pair<u64i, u64i> cmp(const BigInt<t>& other) const { for (int i = t - 1; i >= 0; i--) if (i == 0 || (*this)[i] != other[i]) return {(*this)[i], other[i]}; }
    bool operator< (const BigInt<t>& other) const { auto pr{cmp(other)}; return pr.first <  pr.second; }
    bool operator> (const BigInt<t>& other) const { auto pr{cmp(other)}; return pr.first >  pr.second; }
    bool operator<=(const BigInt<t>& other) const { auto pr{cmp(other)}; return pr.first <= pr.second; }
    bool operator>=(const BigInt<t>& other) const { auto pr{cmp(other)}; return pr.first >= pr.second; }
    bool operator==(const BigInt<t>& other) const { auto pr{cmp(other)}; return pr.first == pr.second; }
    bool operator!=(const BigInt<t>& other) const { auto pr{cmp(other)}; return pr.first != pr.second; }

    // little gagdets
    BigInt<t> half() const 
    {
        BigInt<t> res{*this};
        for (int i = 0; i < t; i++) {
            u64i low{LOW1(res[i])};
            res[i] >>= 1;
            if (i > 0) res[i - 1] |= (low << 63);
        }
        return res;
    }
    u1i lowest_bit() const { return LOW1((*this)[0]); }

    // basic arithmetic operation
    BigInt<t> add_with_carry(const BigInt<t>& other, u1i *c = 0, u1i c_in = 0) const 
    {
        u1i carry{c_in};
        BigInt<t> res{0};
        for (int i = 0; i < t; i++) res[i] = add64_with_carry((*this)[i], other[i], &carry, carry);
        if (c) *c = carry;
        return res;
    }
    BigInt<t> sub_with_carry(const BigInt<t>& other, u1i *c = 0, u1i c_in = 0) const 
    {
        u1i carry{c_in};
        BigInt<t> res{0};
        for (int i = 0; i < t; i++) res[i] = sub64_with_carry((*this)[i], other[i], &carry, carry);
        if (c) *c = carry;
        return res;
    }
    BigInt<2*t> mul(const BigInt<t>& other) const
    {
        u64i r0{0}, r1{0}, r2{0};
        BigInt<2*t> res{0};
        for (int k = 0; k <= 2 * t - 2; k++) {
            int lb, ub;
            if (k <= t - 1) lb = 0, ub = k;
            else lb = k - (t - 1), ub = t - 1;
            for (int i = lb; i <= ub; i++) {
                auto pr{mul64((*this)[i], other[k - i])};
                u1i carry;
                r0 = add64_with_carry(r0, pr.second, &carry);
                r1 = add64_with_carry(r1, pr.first, &carry, carry);
                r2 += carry;
            }
            res[k] = r0, r0 = r1, r1 = r2, r2 = 0;
        }
        res[2 * t - 1] = r0;
        return res;
    }
    BigInt<2*t> square() const
    {
        u64i r0{0}, r1{0}, r2{0};
        BigInt<2*t> res{0};
        for (int k = 0; k <= 2 * t - 2; k++) {
            int lb;
            if (k <= t - 1) lb = 0;
            else lb = k - (t - 1);
            for (int i = lb; i <= (k >> 1); i++) {
                auto pr{mul64((*this)[i], (*this)[k - i])};
                u1i carry;
                if (i < k - i) {
                    pr = left_shift_by_one(pr.first, pr.second, &carry);
                    r2 += carry;
                }
                r0 = add64_with_carry(r0, pr.second, &carry);
                r1 = add64_with_carry(r1, pr.first, &carry, carry);
                r2 += carry;
            }
            res[k] = r0, r0 = r1, r1 = r2, r2 = 0;
        }
        res[2 * t - 1] = r0;
        return res;
    }

    // Barret reduction, mu = [b^t / p] must be pre-calculated, where b = 2^64
    // assuming t = [log_{2^64}(p)] + 1, mu may have (t + 1) b-based bit
    static BigInt<t> reduction(BigInt<2*t> z, BigInt<t> p, BigInt<t+1> mu) 
    {   // return ((u << (t * 64)) | v) % p
        BigInt<t+1> zp{z.val + (t - 1), t + 1}, lp{p.val, t}; // zp := [z / b^{t-1}]
        BigInt<t+3> zpmu{0}; // zpmu := [zp * mu / b^{t-1}]
        {
            u64i r0{0}, r1{0}, r2{0};
            for (int l = t - 1; l <= 2 * t; l++) {
                int lb, ub;
                if (l <= t) lb = 0, ub = l;
                else lb = l - t, ub = t;
                for (int i = lb; i <= ub; i++) {
                    auto pr{mul64(zp[i], mu[l - i])};
                    u1i carry;
                    r0 = add64_with_carry(r0, pr.second, &carry);
                    r1 = add64_with_carry(r1, pr.first, &carry, carry);
                    r2 += carry;
                }
                zpmu[l - (t - 1)] = r0, r0 = r1, r1 = r2, r2 = 0;
            }
            zpmu[t + 2] = r0;
        }
        BigInt<t+1> qhat{zpmu.val + 2, t + 1}, qhatp{0}; // qhat := [zpmu / b^2] = [zp * mu / b^{k+1}], qhatp := (qhat * p) mod b^{t+1}
        {
            u64i r0{0}, r1{0}, r2{0};
            for (int k = 0; k <= t; k++) {
                for (int i = 0; i <= k; i++) {
                    auto pr{mul64(qhat[i], lp[k - i])};
                    u1i carry;
                    r0 = add64_with_carry(r0, pr.second, &carry);
                    r1 = add64_with_carry(r1, pr.first, &carry, carry);
                    r2 += carry;
                }
                qhatp[k] = r0, r0 = r1, r1 = r2, r2 = 0;
            }
        }
        BigInt<t+1> zmod{z.val, t + 1}; // zmod := z mod b^{t+1}
        BigInt<t+1> r{zmod.sub_with_carry(qhatp)};
        while (r >= lp) r = r.sub_with_carry(lp);
        return BigInt<t>{r.val, t};
    }

    // Montgomery reduction, pp = -p^{-1} (mod b^t) must be pre-caculated, where b = 2^64
    static BigInt<t> Montgomery(BigInt<2*t> z, BigInt<t> p, BigInt<t> pp)
    {
        BigInt<t> zpp{0};
        {
            u64i r0{0}, r1{0}, r2{0};
            for (int k = 0; k <= t - 1; k++) {
                for (int i = 0; i <= k; i++) {
                    auto pr{mul64(z[i], pp[k - i])};
                    u1i carry;
                    r0 = add64_with_carry(r0, pr.second, &carry);
                    r1 = add64_with_carry(r1, pr.first, &carry, carry);
                    r2 += carry;
                }
                zpp[k] = r0, r0 = r1, r1 = r2, r2 = 0;
            }
        }
        u1i carry;
        BigInt<2*t> cR{z.add_with_carry(zpp.mul(p), &carry)};
        BigInt<t+1> c{cR.val + t, t}, lp{p.val, t}; c[t] = carry;
        if (c >= lp) c = c.sub_with_carry(lp);
        return BigInt<t>{c.val, t};
    }

    // binary-method for inverse, where p must be a prime (or at least odd with gcd(a, p) = 1)
    static BigInt<t> inverse(BigInt<t> a, BigInt<t> p)
    {
        BigInt<t> u{a}, v{p}, x1{1}, x2{0}, hp{p.half()};
        while (u != BigInt<t>{1} && v != BigInt<t>{1}) {
            while (u.lowest_bit() == 0) {
                u1i odd{x1.lowest_bit()};
                u = u.half(), x1 = x1.half();
                if (odd) x1 = x1.add_with_carry(hp, &odd, odd);
            }
            while (v.lowest_bit() == 0) {
                u1i odd{x2.lowest_bit()};
                v = v.half(), x2 = x2.half();
                if (odd) x2 = x2.add_with_carry(hp, &odd, odd);
            }
            bool neg;
            if (u >= v) {
                u = u.sub_with_carry(v);
                x1 = x1.sub_with_carry(x2, &neg);
                if (neg) x1 = x1.add_with_carry(p);
            }
            else {
                v = v.sub_with_carry(u);
                x2 = x2.sub_with_carry(x1, &neg);
                if (neg) x2 = x2.add_with_carry(p);
            }
        }
        if (u == BigInt<t>{1}) return x1;
        else return x2;
    }
};

typedef BigInt<2> u128i;
typedef BigInt<4> u256i;
typedef BigInt<8> u512i;

enum class ReductionMethod { Barret };

// F_p(:={0, ..., p-1}) field arithmetic
template<size_t t>
class Fp {
    BigInt<t> p, val;
    ReductionMethod rm{ReductionMethod::Barret};
    
    // Barret reduction parameters
    BigInt<t+1> mu{0};
    // Montgomery reduction parameters
    BigInt<t> pp{0}, np{0};
public:
    // p must be a prime but we will not verify
    Fp(BigInt<t> p, BigInt<t> val) : p{p}, val{val} { assert(val < p); }
    void set_reduction_method(ReductionMethod rm_) { rm = rm_; }
    // mu must equal [2^{64*2t} / p] but we will not verify
    void set_Barret_mu(BigInt<t+1> mu_) { assert(rm == ReductionMethod::Barret); mu = mu_; }
    // pp must equal -p^{-1} mod 2^{64*t} bust we will not verify
    void set_Mont_pp(BigInt<t> pp_) { pp = pp_; }

    void print() const { val.print(); }

    // helper function
   BigInt<t> Mont(BigInt<t> xs, BigInt<t> ys) { return BigInt<t>::Montgomery(xs.mul(ys), p, pp); }

    // basic arithmetic operation
    Fp<t> operator+(const Fp<t>& other) const 
    {
        assert(p == other.p);
        u1i carry;
        BigInt<t> res{val.add_with_carry(other.val, &carry)};
        if (carry || res > p) return {p, res.sub_with_carry(p)};
        else return {p, res};
    }
    Fp<t> operator-(const Fp<t>& other) const
    {
        assert(p == other.p);
        u1i carry;
        BigInt<t> res{val.sub_with_carry(other.val, &carry)};
        if (carry) return {p, res.add_with_carry(p)};
        else return {p, res};
    }
    Fp<t> operator*(const Fp<t>& other) const
    {
        assert(p == other.p && mu != BigInt<t+1>{0});
        BigInt<2*t> prod{val.mul(other.val)};
        return {p, BigInt<t>::reduction(prod, p, mu)};
    }
    Fp<t> square() const
    {
        assert(mu != BigInt<t+1>{0});
        BigInt<2*t> prod{val.square()};
        return {p, BigInt<t>::reduction(prod, p, mu)};
    }

    Fp<t> inverse() const { return {p, BigInt<t>::inverse(val, p)}; }
    Fp<t> exponentiation(u1i bit_dec[], size_t l) {
        assert(mu != BigInt<t+1>{0});
        BigInt<2*t> lxs{0}, lA{0}; lA[t] = 1;
        for (int i = 0; i < t; i++) lxs[i + t] = val[i];
        BigInt<t> xs{BigInt<t>::reduction(lxs, p, mu)}, A{BigInt<t>::reduction(lA, p, mu)};
        for (int i = l - 1; i >= 0; i--) {
            A = Mont(A, A);
            if (bit_dec[i] == 1) A = Mont(A, xs);
        }
        return {p, Mont(A, BigInt<t>{1})};
    }
};

typedef Fp<2> fp128;
typedef Fp<4> fp256;
typedef Fp<8> fp512;

int main(int argc, char *argv[])
{
    u64i p_arr[4]  = {0xffffffffffffffff, 0x00000000ffffffff, 0x0000000000000000, 0xffffffff00000001};
    u64i pp_arr[4] = {0x0000000000000001, 0x0000000100000000, 0x0000000000000000, 0xffffffff00000002};
    u64i mu_arr[5] = {0x0000000000000003, 0xfffffffeffffffff, 0xfffffffefffffffe, 0x00000000ffffffff, 0x0000000000000001};
    fp256 a{u256i{p_arr, 4}, u256i::sample()}, b{u256i{p_arr, 4}, u256i::sample()};
    a.set_Barret_mu(BigInt<5>{mu_arr, 5}), b.set_Barret_mu(BigInt<5>{mu_arr, 5});
    a.set_Mont_pp(BigInt<4>{pp_arr, 4}), b.set_Mont_pp(BigInt<4>{pp_arr, 4});

    a.print(), b.print();
    (a + b).print();
    (a - b).print();
    (a * b).print();
    a.square().print();
    cout << endl;

    fp256 c = a.inverse();
    c.print();
    (a * c).print();

    u1i e[4] = {0, 0, 0, 1};
    b = a.square(); b.set_Barret_mu(BigInt<5>{mu_arr, 5}); b.set_Mont_pp(BigInt<4>{pp_arr, 4});
    b = b.square(); b.set_Barret_mu(BigInt<5>{mu_arr, 5}); b.set_Mont_pp(BigInt<4>{pp_arr, 4});
    b = b.square(); b.set_Barret_mu(BigInt<5>{mu_arr, 5}); b.set_Mont_pp(BigInt<4>{pp_arr, 4});
    b.print();
    a.exponentiation(e, 4).print();

    return 0;
}