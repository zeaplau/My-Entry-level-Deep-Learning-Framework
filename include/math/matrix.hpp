#pragma once
#include <atomic>
#include <corecrt.h>
#include <cstddef>
#include <cassert>
#include <float.h>
#include <functional>
#include <thread>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vcruntime.h>
#include <vector>
#include <memory>
#include <numeric>

namespace dddl {
    template<typename MType>
    class Matrix {
        public:
            using mdata_ptr = std::shared_ptr<std::vector<MType>>;
            using mdata = std::vector<MType>;
            using m_dim = std::vector<unsigned long>;
            using slice_param = std::vector<unsigned long>;
            using ul_pos = std::pair<unsigned long, unsigned long>;

            Matrix();
            Matrix(const Matrix<MType>& m);
            ~Matrix();

            Matrix(m_dim shape, MType fill_with);
            Matrix(m_dim shape, mdata_ptr data); // 1d-array
            Matrix(m_dim shape, std::initializer_list<MType> init_data);
            Matrix(std::initializer_list<unsigned long> shape, MType fill_with);
            Matrix(std::initializer_list<unsigned long> shape, mdata_ptr data);
            Matrix(std::initializer_list<unsigned long> shape, std::initializer_list<unsigned long> init_data);

            // Operator
            Matrix<MType>& operator=(const Matrix<MType>& m);
            Matrix<MType>& operator+(const Matrix<MType>& m);
            Matrix<MType>& operator+(MType number);
            Matrix<MType>& operator*(MType weight);
            Matrix<MType>& operator*(const Matrix<MType>& m);
            Matrix<MType>& operator+=(const Matrix<MType>& m);
            Matrix<MType>& operator+=(MType number);
            Matrix<MType>& operator*=(const Matrix<MType>& m);
            Matrix<MType>& operator*=(MType weight);
            bool operator==(const Matrix<MType>& m);
            void copy_from(const Matrix<MType>& m);

            Matrix<MType>& add(const Matrix<MType>& addend); // inplace
            Matrix<MType>& add(MType number);
            Matrix<MType>& mul(const Matrix<MType>& multiplier); // inplace
            Matrix<MType>& mul_v(const Matrix<MType>& multipier); // inplace
            Matrix<MType>& scale(MType weight);

            void clear_data();
            void set_data(mdata_ptr data);
            void resize(m_dim shape, MType filled_with);
            void resize(unsigned long n, unsigned long c, unsigned long h, unsigned long w, MType filled_with);
            void view(m_dim shape);
            void view(unsigned long n, unsigned long c, unsigned long h, unsigned long w);
            void view(std::initializer_list<unsigned long> shape);
            void concat(Matrix<MType>& m, unsigned long dim);
            Matrix<MType> sum_by_dim(unsigned long sum_dim);
            Matrix<MType> slice(std::initializer_list<unsigned long> slice_index);
            Matrix<MType> slice(slice_param slice_index);
            Matrix<MType> slice(unsigned long slice_dim, unsigned long start, unsigned long end);

            void T();
            bool is_uninitialized();
            void check_initialized() const;
            auto get(unsigned long n, unsigned long c, unsigned long h, unsigned long w) -> MType;
            auto get(unsigned long index) -> MType;
            void set(unsigned long, unsigned long c, unsigned long h, unsigned long w, MType value);
            void set(unsigned long index, MType value);
            auto get_batch(m_dim dim, unsigned long batch_id) const -> ul_pos;
            auto get_batch(unsigned long batch_id) const -> ul_pos;
            auto get_channal(m_dim dim, unsigned long channal_id, ul_pos batch_pos) const -> ul_pos;
            auto get_channal(unsigned long channal_id, ul_pos batch_pos) const -> ul_pos;
            
            auto get_data() const -> const mdata_ptr;
            auto get_dim() const -> const m_dim;
            auto get_mdata() -> mdata_ptr;
        private:
            void mul_core(const mdata_ptr inplace_m, ul_pos inplace_m_pos, m_dim inplace_m_dim, const mdata_ptr mul_m, ul_pos mul_m_pos, m_dim mul_m_dim, mdata_ptr result, ul_pos result_pos);
            void add_boardcast_core(Matrix<MType>& summand, const Matrix<MType>& addend, ul_pos channal_ul, ul_pos add_channal_ul, int piece);
            void mul_v_boardcast_core(Matrix<MType>& multiplied, const Matrix<MType>& multiplier, ul_pos channal_ul, ul_pos mul_channal_ul, int piece);
            void T_core(ul_pos channal_ul);
            unsigned long T_core_get_next(unsigned long now_index, unsigned long m_h, unsigned long m_len);
            void sum_by_dim_core(Matrix<MType>& m, Matrix<MType>& result, unsigned long dim, unsigned long batch_id);
            void concat_core(Matrix<MType>& m, Matrix<MType>& result, unsigned long dim, unsigned long batch_id);
            mdata_ptr data;
            m_dim shape; // (n, c, h, w)
            bool uninitialized {false};
    };

    template<typename MType>
    Matrix<MType>::Matrix() {
        uninitialized = true;
        data = std::make_shared<mdata>(0, MType(0));
        shape = m_dim{0, 0, 0, 0};
    }

    template<typename MType>
    Matrix<MType>::Matrix(const Matrix<MType>& m) {
        data = m.data;
        shape = m.shape;
    }

    template<typename MType>
    Matrix<MType>::Matrix(m_dim shape, MType fill_with) {
        assert(shape.size() == 4);
        long data_len = 1;
        for (auto dim : shape) {
            assert(dim >= 1);
            data_len *= dim;
        }
        data = std::make_shared<mdata>(data_len, fill_with);
        this->shape = shape;
    }

    template<typename MType>
    Matrix<MType>::Matrix(m_dim shape, mdata_ptr data) {
        assert(shape.size() == 4);
        long data_len = 1;
        for (auto dim : shape) {
            assert(dim >= 1);
            data_len *= dim;
        }
        assert(data_len == data->size());
        this->data = data;
        this->shape = shape;
    }

    template<typename MType>
    Matrix<MType>::Matrix(m_dim shape, std::initializer_list<MType> init_data) {
        assert(shape[0] * shape[1] * shape[2] * shape[3] == init_data.size());
        this->shape = shape;
        data = std::make_shared<mdata_ptr>(init_data);
    }

    template<typename MType>
    Matrix<MType>::Matrix(std::initializer_list<unsigned long> shape, MType fill_with) {
        assert(shape.size() == 4);
        new (this)Matrix<MType>(m_dim {shape}, fill_with);
    }

    template<typename MType>
    Matrix<MType>::Matrix(std::initializer_list<unsigned long> shape, mdata_ptr data) {
        assert(shape.size() == 4);
        new (this)Matrix<MType>(m_dim {shape}, data);
    }

    template<typename MType>
    Matrix<MType>::Matrix(std::initializer_list<unsigned long> shape, std::initializer_list<unsigned long> init_data) {
        assert(shape.size() == 4);
        new (this)Matrix<MType>(m_dim {shape}, init_data);
    }

    template<typename MType>
    Matrix<MType>& Matrix<MType>::operator=(const Matrix<MType> &m) {
        if (uninitialized) {
            this->data = m.data;
            this->shape = m.shape;
            uninitialized = false;
            return *this;
        }

        if (*this == m) {
            return *this;
        } else {
            this->data = m.data;
            this->shape = m.shape;
        }
        uninitialized = false;
        return *this;
    }

    template<typename MType>
    Matrix<MType>& Matrix<MType>::operator+(const Matrix<MType>& m) {
        add(m);
        return *this;
    }

    template<typename MType>
    Matrix<MType>& Matrix<MType>::operator+(MType number) {
        add(number);
        return *this;
    }

    template<typename MType>
    Matrix<MType>& Matrix<MType>::operator*(const Matrix<MType>& m) {
        check_initialized();
        mul(m);
        return *this;
    }

    template<typename MType>
    Matrix<MType>& Matrix<MType>::operator*(MType weight) {
        scale(weight);
        return *this;
    }

    template<typename MType>
    Matrix<MType>& Matrix<MType>::operator+=(const Matrix<MType>& m) {
        add(m);
        return *this;
    }

    template<typename MType>
    Matrix<MType>& Matrix<MType>::operator+=(MType number) {
        add(number);
        return *this;
    }

    template<typename MType>
    Matrix<MType>& Matrix<MType>::operator*=(const Matrix<MType>& m) {
        mul(m);
        return *this;
    }

    template<typename MType>
    Matrix<MType>& Matrix<MType>::operator*=(MType weight) {
        scale(weight);
        return *this;
    }

    template<typename MType>
    bool Matrix<MType>::operator==(const Matrix<MType> &m) {
        check_initialized();
        if (this->shape != m.shape) {
            return false;
        }
        if (this->data != m.data) {
            return false;
        }
        return true;
    }

    template<typename MType>
    void Matrix<MType>::copy_from(const Matrix<MType> &m) {
        shape = m.shape;
        data->resize(
            std::accumulate(shape.begin(), shape.end(), 0)
        );
        for (size_t i = 0; i < data.size(); i++) {
            data->at(i) = m.data->at(i);
        }
        uninitialized = false;
    }

    template<typename MType>
    Matrix<MType>& Matrix<MType>::add(const Matrix<MType>& addend) {
        check_initialized();
        // (n, c, h, w)
        assert(addend.shape[0] == shape[0] && addend.shape[1] == shape[1]);
        // if match
        if (this->data->size() == (addend.data)->size()) {
            for (size_t i = 0; i < this->data->size(); i++) {
                data->at(i) += (addend.data)->at(i);
            }
        } else if (this->data->size() % (addend.data)->size() == 0 && (this->data->size() > (addend.data)->size())) {
            // if this.shape bigger than addend one
            std::vector<std::thread> threads;
            unsigned long pieces = data->size() / addend.data->size();
            // each n
            for (unsigned long i = 0; i < shape[0]; i++) {
                ul_pos this_bul = get_batch(i);
                ul_pos addend_bul = addend.get_batch(i);
                for (unsigned long c = 0; c < shape[1]; c++) {
                    ul_pos this_cul = get_channal(c);
                    ul_pos addend_cul = addend.get_channal(c);
                    for (unsigned long p = 0; p < pieces; p++) {
                        threads.emplace_back(&Matrix<MType>::add_boardcast_core, this, std::ref(*this), addend, this_cul, addend_cul, p);
                    }
                }
            }
            for (auto& t : threads) {
                t.join();
            }
        } else if ((addend.data)->size() % this->data->size() == 0 && ((addend.data)->size() > this->data->size())) {
            // if addend one bigger than this.shape
            Matrix<MType> bigger {addend};
            std::vector<std::thread> threads;
            unsigned long pieces = data->size() / addend.data->size();
            for (unsigned long i = 0; i < shape[0]; i++) {
                ul_pos this_bul = get_batch(i);
                ul_pos addend_bul = get_batch(i);
                for (unsigned long c = 0; c < shape[1]; c++) {
                    ul_pos this_cul = get_channal(c);
                    ul_pos addend_cul = get_channal(c);
                    for (unsigned long p = 0; p < pieces; p++) {
                        threads.emplace_back(&Matrix<MType>::add_boardcast_core, this, std::ref(bigger), *this, 
                        this_cul, addend_cul, p);
                    }
                }
            }
            for (auto& t : threads) {
                t.join();
            }
        } else {
            throw std::runtime_error("Shape is not match.");
        }
        return *this;
    }

    template<typename MType>
    Matrix<MType>& Matrix<MType>::add(MType number) {
        check_initialized();
        for (size_t i = 0; i < data->size(); i++) {
            data->at(i) += number;
        }
    }

    /*
        mul: matmul
    */
    template<typename MType>
    void Matrix<MType>::mul_core(const mdata_ptr inplace_m, ul_pos inplace_m_pos, m_dim inplace_m_dim, const mdata_ptr mul_m, ul_pos mul_m_pos, m_dim mul_m_dim, mdata_ptr result, ul_pos result_pos) {
        assert(inplace_m_dim.size() == 2 && mul_m_dim.size() == 2);
        assert(inplace_m_dim[0] == mul_m_dim[1] && inplace_m_dim[1] == mul_m_dim[0]);
        assert(result->size() % (inplace_m_dim[0] * mul_m_dim[1]) == 0);

        // (h, w) * (w, h) => (h, h)
        unsigned long rh, rw, i;
        MType sum;
        for (rh = 0; rh < inplace_m_dim[0]; rh++) {
            for (rw = 0; rw < mul_m_dim[1]; rw++) {
                sum = MType(0);
                for (i = 0; i < inplace_m_dim[1]; i++) {
                    // rh * inplace_m_dim[1] => previous tensor element
                    // i => exact location
                    sum += inplace_m->at(inplace_m_pos.first + rh * inplace_m_dim[1] + i) * 
                        mul_m->at(mul_m_pos.first + i * inplace_m_dim[0] + rw);
                }
                result->at(result_pos.first + rh * mul_m_dim[1] + rw) = sum;
            }
        }
    }

    template<typename MType>
    Matrix<MType>& Matrix<MType>::mul(const Matrix<MType> &multiplier) {
        check_initialized();
        assert(shape[0] == multiplier.shape[0] && shape[1] == multiplier.shape[1]);
        mdata_ptr mul_res {std::make_shared<mdata>()};
        mul_res->resize(shape[0] * shape[1] * shape[2] * multiplier.shape[3], 0);
        m_dim mul_dim {shape[0], shape[1], shape[2], multiplier.shape[3]};
        std::vector<std::thread> threads;

        // (n, c, h, w)
        for (unsigned long n = 0; n < shape[0]; n++) {
            ul_pos this_bul = get_batch(shape, n);
            ul_pos mul_bul = get_batch(multiplier.shape, n);
            for (unsigned long c = 0; c < shape[1]; c++) {
                ul_pos this_cul = get_channal(shape, c, this_bul);
                ul_pos mul_cul = get_channal(shape, c, mul_cul);
                m_dim this_dim {shape[2], shape[3]};
                m_dim multi_dim {multiplier.shape[2], multiplier.shape[3]};

                ul_pos res_bul = get_batch(multi_dim, n);
                ul_pos res_cul = get_channal(multi_dim, c, res_bul);
                // run on `this` object
                threads.push_back(std::thread {&Matrix<MType>::mul_core, this, static_cast<const mdata_ptr>(data), this_cul, this_dim, static_cast<const mdata_ptr>(multiplier.data), mul_cul, multi_dim, mul_res, res_cul});
            }
        }
        for (auto& t : threads) {
            t.join();
        }
        data.reset();
        data = mul_res;
        shape = mul_dim;
        return *this;
    }

    /*
        mul_v: element-wise
    */
    template<typename MType>
    Matrix<MType>& Matrix<MType>::mul_v(const Matrix<MType> &multipier) {
        check_initialized();
        // same size
        if (data->size() == multipier->size()) {
            for (unsigned long i = 0; i < data->size(); i++) {
                data->at(i) *= multipier->at(i);
            }
        } else if (data->size() % multipier->size() == 0 && data->size() >= multipier->size()) {
            std::vector<std::thread> threads;
            size_t pieces = data->size() / multipier->size();
            for (unsigned long n = 0; n < shape[0]; n++) {
                ul_pos this_bul = get_batch(n);
                ul_pos mul_bul = multipier.get_batch(n);
                for (unsigned long c = 0; c < shape[1]; c++) {
                    ul_pos this_cul = get_channal(c);
                    ul_pos mul_cul = multipier.get_channal(c);
                    for (unsigned long p = 0; p < pieces; p++) {
                        threads.emplace_back(&Matrix<MType>::mul_v_boardcast_core, this, std::ref(*this), multipier, this_cul, mul_cul, p);
                    }
                }
            }
            for (auto& t : threads) {
                t.join();
            }
        } else if (multipier->size() % data->size() == 0 && multipier->size() >= data->size()) {
            Matrix<MType> bigger {multipier};
            std::vector<std::thread> threads;
            size_t pieces = bigger->size() / data->size();
            for (unsigned long n = 0; n < shape[0]; n++) {
                ul_pos this_bul = get_batch(n);
                ul_pos mul_bul = bigger.get_batch(n);
                for (unsigned long c = 0; c < shape[1]; c++) {
                    ul_pos this_cul = get_channal(c);
                    ul_pos mul_cul = multipier.get_channal(c);
                    for (unsigned long p = 0; p < pieces; p++) {
                        threads.emplace_back(&Matrix<MType>::mul_v_boardcast_core, this, std::ref(bigger), *this, this_cul, mul_cul, p);
                    }
                }
            }
            for (auto& t : threads) {
                t.join();
            }
        } else {
            throw std::runtime_error("Shape is not match.");
        }
        return *this;
    }

    template<typename MType>
    void Matrix<MType>::add_boardcast_core(Matrix<MType>& summand, const Matrix<MType>& addend, ul_pos channal_ul, ul_pos add_channal_ul, int piece) {
        assert(summand.data->size() > addend.data->size());
        assert(summand.data->size() % addend.data->size() == 0);
        // how many pieces
        size_t pieces = summand.data->size() / addend.data->size();
        assert(pieces > 0);
        m_dim addend_dim {addend.get_dim()};
        assert(summand.shape[2] % addend_dim[2] == 0);
        assert(summand.shape[3] % addend_dim[2] == 0);

        // (n, c, h, w)
        unsigned long pw, px, py;
        // (h, w) -> pw * (px, py)
        pw = summand.shape[3] / addend_dim[3];
        px = piece % pw;
        py = piece / pw;
        for (unsigned long ah = 0; ah < addend_dim[2]; ah++) {
            for (unsigned long aw = 0; aw < addend_dim[3]; aw++) {
                // (x, y) in summand
                summand.data->at(
                    channal_ul.first + 
                    (py * addend_dim[2] + ah) * summand.shape[2] + 
                    px * addend_dim[3] + aw) += 
                // (x, y) in addend
                addend.data->at(
                    add_channal_ul.first + 
                    ah * addend_dim[2] + 
                    aw);
            }
        }
    }

    template<typename MType>
    void Matrix<MType>::mul_v_boardcast_core(Matrix<MType>& multiplied, const Matrix<MType>& multiplier, ul_pos channal_ul, ul_pos mul_channal_ul, int piece) {
        assert(multiplied.data->size() > multiplier.data->size());
        assert(multiplied.data->size() % multiplier.data->size() == 0);

        size_t pieces = multiplied.data->size() / multiplier.data->size();
        auto multiplier_dim = multiplier.get_dim();
        assert(multiplied.shape[2] % multiplier_dim[2] == 0);
        assert(multiplied.shape[3] % multiplier_dim[3] == 0);
        // (n, c, h, w)
        unsigned long pw, px, py;
        pw = multiplied.shape[3] / multiplier_dim[3];
        px = multiplied.shape[3] % pw;
        py = multiplied.shape[3] / pw;
        for (unsigned long mh = 0; mh < multiplier_dim[2]; mh++) {
            for (unsigned long mw = 0; mw < multiplier_dim[3]; mw++) {
                multiplied.data->at(
                    channal_ul.first + 
                    (py * multiplier_dim[2] + mh) * multiplied.shape[2] + 
                    (px * multiplier_dim[3] + mw)
                ) *= 
                multiplier.data->at(
                    mul_channal_ul.first + mh * multiplier_dim[2] + mw
                );
            }
        }
    }

    template<typename MType>
    Matrix<MType>& Matrix<MType>::scale(MType weight) {
        check_initialized();
        for (size_t i = 0; i < data->size(); i++) {
            data->at(i) *= weight;
        }
        return *this;
    }

    template<typename MType>
    void Matrix<MType>::clear_data() {
        uninitialized = true;
        data->resize(0, 0);
        shape = m_dim {0, 0, 0, 0};
    }

    template<typename MType>
    void Matrix<MType>::set_data(mdata_ptr data) {
        if (this->data == data) {
            return ;
        }
        this->data = data;
        uninitialized = false;
    }

    template<typename MType>
    void Matrix<MType>::resize(m_dim shape, MType filled_with) {
        assert(shape.size() == 4);
        assert(shape[0] >= 0 && shape[1] >= 0 && shape[2] >= 0 && shape[3] == 0);
        if (this->shape == shape) {
            return ;
        }
        this->shape = shape;
        data->resize(std::accumulate(shape.begin(), shape.end(), 0), filled_with);
        uninitialized = false;
    }

    template<typename MType>
    void Matrix<MType>::resize(unsigned long n, unsigned long c, unsigned long h, unsigned long w, MType filled_with) {
        assert(n >= 1);
        assert(c >= 1);
        assert(h >= 1);
        assert(w >= 1);
        m_dim new_dim {n, c, h, w};
        resize(new_dim, filled_with);
    }

    template<typename MType>
    void Matrix<MType>::view(m_dim shape) {
        assert(shape.size() == 4);
        assert(shape[0] >= 1 && shape[1] >= 1 && shape[2] >= 1 && shape[3] >= 1);
        assert(
            std::accumulate(shape.begin(), shape.end(), std::multiplies<unsigned long>())  == std::accumulate(this->shape.begin(), this->shape.end(), std::multiplies<unsigned long>())
        );
        this->shape = shape;
    }

    template<typename MType>
    void Matrix<MType>::view(unsigned long n, unsigned long c, unsigned long h, unsigned long w) {
        view(m_dim {n, c, h, w});
    }

    template<typename MType>
    void Matrix<MType>::view(std::initializer_list<unsigned long> shape) {
        view(m_dim {shape});
    }

    template<typename MType>
    void Matrix<MType>::concat(Matrix<MType> &m, unsigned long dim) {
        check_initialized();
        assert(dim > 0 && dim <= 3);
        m_dim res_dim {shape};
        res_dim += m.shape[dim];
        Matrix<MType> res {res_dim, MType(0)};
        std::vector<std::thread> threads;
        for (unsigned long n = 0; n < shape[0]; n++) {
            threads.emplace_back(&Matrix<MType>::concat_core, this, std::ref(m), std::ref(res), dim, n);
        }
        for (auto& t : threads) {
            t.join();
        }
        this->copy_from(res);
    }

    template<typename MType>
    void Matrix<MType>::concat_core(Matrix<MType>& m, Matrix<MType>& result, unsigned long dim, unsigned long batch_id) {
        auto checkValid = [=](unsigned long check_dim, Matrix<MType>& m) {
            for (unsigned long dim = 0; dim < m.shape.size(); dim++) {
                if (dim != check_dim) {
                    assert(this->shape[dim] == m.shape[dim]);
                }
            }
        };
        // (n, c, h, w)
        checkValid(dim, m);
        if (dim == 1) {
            for (unsigned long c = 0; c < shape[1]; c++) {
                for (unsigned long h = 0; h < shape[2]; h++) {
                    for (unsigned long w = 0; w < shape[3]; w++) {
                        result.set(batch_id, c, h, w, this->get(batch_id, c, w));
                    }
                }
            }
            for (unsigned long ic = 0; ic < m.shape[1]; ic++) {
                for (unsigned long h = 0; h < m.shape[2]; h++) {
                    for (unsigned long w = 0; w < m.shape[3]; w++) {
                        result.set(batch_id, ic + m.shape[1], h, w, m.get(batch_id, ic, h, w));
                    }
                }
            }
        } else if (dim == 2) {
            for (unsigned long c = 0; c < shape[1]; c++) {
                for (unsigned long ih = 0; ih < shape[2]; ih++) {
                    for (unsigned long w = 0; w < shape[3]; w++) {
                        result.set(batch_id, c, ih, w, this->get(batch_id, c, ih, w));
                    }
                }
                for (unsigned long ih = 0; ih < m.shape[2]; ih++) {
                    for (unsigned long w = 0; w < m.shape[3]; w++) {
                        result.set(batch_id, c, ih + shape[2], w, m.get(batch_id, c, ih, w));
                    }
                }
            }
        } else if (dim == 3) {
            for (unsigned long c = 0; c < shape[1]; c++) {
                for (unsigned long h = 0; h < shape[2]; h++) {
                    for (unsigned long iw = 0; iw < shape[3]; iw++) {
                        result.set(batch_id, c, h, iw, this->get(batch_id, c, h, iw));
                    }
                    for (unsigned long iw = 0; iw < m.shape[3]; iw++) {
                        result.set(batch_id, c, h, iw + shape[3], this->get(batch_id, c, h, iw));
                    }
                }
            }
        }
    }

    template<typename MType>
    Matrix<MType> Matrix<MType>::sum_by_dim(unsigned long sum_dim) {
        check_initialized();
        // sum of all element
        if (sum_dim == -1) {
            auto m_sum = MType(0);
            m_dim new_shape = {1, 1, 1, 1};
            for (size_t i = 0; i < data->size(); i++) {
                m_sum += data->at(i);
            }
            return Matrix<MType> {new_shape, m_sum};
        }
        // sum in dim
        assert(sum_dim >= 1 && sum_dim < 4);
        m_dim new_shape {shape};
        new_shape[sum_dim] = 1;
        Matrix<MType> new_matrix {new_shape, 0};
        for (unsigned long n = 0; n < shape[0]; n++) {
            sum_by_dim_core(*this, new_matrix, sum_dim, n);
        }
        return new_matrix;
    }

    template<typename MType>
    void Matrix<MType>::sum_by_dim_core(Matrix<MType>& m, Matrix<MType>& result, unsigned long dim, unsigned long batch_id) {
        // (n, c, h, w)
        if (dim == 1) {
            for (unsigned long ih = 0; ih < m.shape[2]; ih++) {
                for (unsigned long iw = 0; iw < m.shape[3]; iw++) {
                    MType sum_value = 0;
                    for (unsigned long ic = 0; ic < m.shape[1]; ic++) {
                        sum_value += m.get(batch_id, ic, ih, iw);
                    }
                }
            }
        } else if (dim == 2) {
            for (unsigned long ic = 0; ic < m.shape[1]; ic++) {
                for (unsigned long iw = 0; iw < m.shape[3]; iw++) {
                    MType sum_value = 0;
                    for (unsigned long ih = 0; ih < m.shape[2]; ih++) {
                        sum_value += m.get(batch_id, ic, ih, iw);
                    }
                }
            }
        } else if (dim == 3) {
            for (unsigned long ic = 0; ic < m.shape[1]; ic++) {
                for (unsigned long ih = 0; ih < m.shape[2]; ih++) {
                    MType sum_value = 0;
                    for (unsigned long iw = 0; iw < m.shape[3]; iw++) {
                        sum_value += m.get(batch_id, ic, ih, iw);
                    }
                }
            }
        } else {
            return ;
        }
    }

    template<typename MType>
    Matrix<MType> Matrix<MType>::slice(std::initializer_list<unsigned long> slice_index) {
        return slice(slice_param {slice_index});
    }

    template<typename MType>
    Matrix<MType> Matrix<MType>::slice(slice_param slice_index) {
        check_initialized();
        // index: n[srart, end], c[start, end], h[start, end], w[start, end]
        assert(slice_index.size() == 8);
        for (size_t si = 0; si < 8; si++) {
            if (slice_index[si] < 0) {
                slice_index[si] += shape[si / 2];
            }
            if (slice_index[si] > shape[si / 2]) {
                throw std::runtime_error("Slice index out of range.");
            }
        }

        m_dim dims {0, 0, 0, 0};
        for (size_t si = 0; si < 8; si++) {
            if (slice_index[si + 1] < slice_index[si]) {
                throw std::runtime_error("Slice range invalid.");
            }
            dims[si / 2] = slice_index[si + 1] - slice_index[si];
        }
        Matrix<MType> res {dims, MType(0)};
        unsigned long ri = 0;
        for (unsigned long n = slice_index[0]; n < slice_index[1]; n++) {
            for (unsigned long c = slice_index[2]; c < slice_index[3]; c++) {
                for (unsigned long h = slice_index[4]; h < slice_index[5]; h++) {
                    for (unsigned long w = slice_index[6]; w < slice_index[7]; w++) {
                        res.set(ri++, this->get(n, c, h, w));
                    }
                }
            }
        }
        return res;
    }

    template<typename MType>
    Matrix<MType> Matrix<MType>::slice(unsigned long slice_dim, unsigned long start, unsigned long end) {
        check_initialized();
        slice_param dims {0, 0, 0, 0, 0, 0, 0, 0};
        dims[slice_dim * 2] = start;
        dims[slice_dim * 2 + 1] = end;
        for (size_t i = 0; i < dims.size(); i++) {
            if (i == slice_dim) {
                continue;
            }
            dims[i * 2 + 1] = shape[i / 2];
        }
        return slice(dims);
    }

    template<typename MType>
    void Matrix<MType>::T() {
        check_initialized();
        std::vector<std::thread> threads;
        // (n, c, h, w)
        for (unsigned long n = 0; n < shape[0]; n++) {
            ul_pos batch_ul = get_batch(n);
            for (unsigned long c = 0; c < shape[1]; c++) {
                ul_pos channal_ul = get_channal(c);
                threads.emplace_back(&Matrix<MType>::T_core, this, channal_ul);
            }
            for (auto& t : threads) {
                t.join();
            }
        }
        resize(shape[0], shape[1], shape[3], shape[2], MType(0));
    }

    template<typename MType>
    void Matrix<MType>::T_core(ul_pos channal_ul) {
        auto get_next = [](unsigned long idx, unsigned long m_h, unsigned long m_len) {
            return (idx * m_h) % (m_len - 1);
        };

        // index -> (index * h) % (len - 1)
        unsigned long m_len = shape[2] * shape[3];
        for (unsigned long i = 0; i < m_len; i++) {
            unsigned long ni = get_next(i, shape[2], m_len);
            while (i < ni) {
                ni = get_next(ni, shape[2], m_len);
            }
            if (i == ni) {
                unsigned long n = get_next(i, shape[2], m_len);
                MType pre = data->set(channal_ul.first + i);
                MType tmp;
                while (i != n) {
                    tmp = data->at(channal_ul.first + n);
                    data->at(channal_ul.first + n) = pre;
                    pre = tmp;
                    n = get_next(n, shape[2], m_len);
                }
                data->at(channal_ul.first + n) = pre;
            }
        }
    }

    template<typename MType>
    bool Matrix<MType>::is_uninitialized() {
        return uninitialized;
    }

    template<typename MType>
    void Matrix<MType>::check_initialized() const {
        if (uninitialized) {
            throw  std::runtime_error("Matrix is uninitialized.");
        }
    }

    template<typename MType>
    MType Matrix<MType>::get(unsigned long n, unsigned long c, unsigned long h, unsigned long w) {
        check_initialized();
        auto isOk = [](int num, int dim, m_dim& shape) {
            assert(num > 0 && num < shape[dim]);
        };
        isOk(n, 0, shape);
        isOk(c, 1, shape);
        isOk(h, 2, shape);
        isOk(w, 3, shape);
        return data->at(
            n * shape[1] * shape[2] * shape[3] + 
            c * shape[2] * shape[3] + 
            h * shape[3] + 
            w
        );
    }

    template<typename MType>
    MType Matrix<MType>::get(unsigned long index) {
        assert(index >= 0 && index < data->size());
        return data->at(index);
    }

    template<typename MType>
    void Matrix<MType>::set(unsigned long n, unsigned long c, unsigned long h, unsigned long w, MType value) {
        check_initialized();
        auto isOk = [](int num, int dim, m_dim& shape) {
            assert(num > 0 && num < shape[dim]);
        };
        isOk(n, 0, shape);
        isOk(c, 1, shape);
        isOk(h, 2, shape);
        isOk(w, 3, shape);
        data->at(
            n * shape[1] * shape[2] * shape[3] + 
            c * shape[2] * shape[3] + 
            h * shape[3] + 
            w
        ) = value;
    }

    template<typename MType>
    void Matrix<MType>::set(unsigned long index, MType value) {
        check_initialized();
        assert(index < data->size());
        data->at(index) = value;
    }

    template<typename MType>
    typename Matrix<MType>::ul_pos Matrix<MType>::get_batch(m_dim dim, unsigned long batch_id) const {
        check_initialized();
        // (n, c, h, w)
        assert(batch_id < dim[0]);
        unsigned long batch_len = std::accumulate(dim.begin() + 1, dim.end(), 0);
        std::pair<int, int> pos {batch_id * batch_id, (batch_id + 1) * batch_len - 1};
        return static_cast<const ul_pos>(pos);
    }

    template<typename MType>
    typename Matrix<MType>::ul_pos Matrix<MType>::get_batch(unsigned long batch_id) const {
        check_initialized();
        return get_batch(shape, batch_id);
    }

    template<typename MType>
    typename Matrix<MType>::ul_pos Matrix<MType>::get_channal(m_dim dim, unsigned long channal_id, ul_pos batch_pos) const {
        check_initialized();
        assert(channal_id < dim[1]);

        // (n, c, h, w)
        unsigned long channal_len = dim[2] * dim[3];
        std::pair<int, int> pos {channal_id * channal_len + batch_pos.first, (channal_id + 1) * channal_len + batch_pos.first - 1};
        assert(pos.second <= batch_pos.second);
        return static_cast<ul_pos>(pos);
    }

    template<typename MType>
    typename Matrix<MType>::ul_pos Matrix<MType>::get_channal(unsigned long channal_id, ul_pos batch_pos) const {
        check_initialized();
        return get_channal(shape, channal_id, batch_pos);
    }

}