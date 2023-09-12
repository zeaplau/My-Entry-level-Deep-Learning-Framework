#pragma once

#include "matrix.hpp"
#include <cmath>
#include <initializer_list>
#include <memory.h>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>
#include <memory>
#include <random>

namespace dddl {
    namespace matrix_tools {
        template<typename MType>
        class MakeMatrix {
            public:
                using mdata_ptr = std::shared_ptr<std::vector<MType>>;
                using m_dim = std::vector<unsigned long>;
                using kernel_shape = std::vector<unsigned long>;
                using ul_pos = std::pair<int, int>;

                MakeMatrix() {};
                MakeMatrix(const m_dim init) : dim(init) {};
                MakeMatrix(std::initializer_list<unsigned long> init) : dim(init) {};

                // init weights
                void gaussian(Matrix<MType>& m);
                void uniform(Matrix<MType>& m);
                void xavier(Matrix<MType>& m, double gain);
                void kaiming(Matrix<MType>& m, double gain);

                // init matrix
                void diagonal(Matrix<MType>& m, MType fill_with);
                void diagonal(Matrix<MType>& m, const Matrix<MType>& fill_with);
                void special_jacobi(Matrix<MType>& m, const Matrix<MType>& sj_fw, unsigned long jk);
                void zeros(Matrix<MType>& m);
                void ones(Matrix<MType>& m);
                void identify(Matrix<MType>& m);

                // padding
                void add_padding(Matrix<MType>& m, Matrix<MType>& result, unsigned long padding, MType fill_with);
                void add_padding(Matrix<MType>& m, Matrix<MType>& result, kernel_shape padding, MType fill_with);
                void add_padding(Matrix<MType>& m, Matrix<MType>& result, std::initializer_list<unsigned long> padding, MType fill_with);
                void sub_padding(Matrix<MType>& m, Matrix<MType>& result, unsigned long padding);
                void sub_padding(Matrix<MType>& m, Matrix<MType>& result, kernel_shape padding);
                void sub_padding(Matrix<MType>& m, Matrix<MType>& result, std::initializer_list<unsigned long> padding);

                // convert
                void img2col(Matrix<MType>& m, Matrix<MType>& result, kernel_shape kernel_size, unsigned long stride);
                void img2col(Matrix<MType>& m, Matrix<MType>& result, std::initializer_list<unsigned long> kernel_size, unsigned long stride);
                void img2col(Matrix<MType>& m, Matrix<MType>& result, unsigned long kernel_size, unsigned long stride);
                void col2img(Matrix<MType>& m, Matrix<MType>& result, unsigned long kernel_size, unsigned long stride, m_dim fw);
                void col2img(Matrix<MType>& m, Matrix<MType>& result, kernel_shape kernel_size, unsigned long stride, m_dim fw);
                void col2img(Matrix<MType>& m, Matrix<MType>& result, std::initializer_list<unsigned long> kernel_size, unsigned long stride, m_dim fw);
                
                // dim change
                void modify_dim(m_dim new_dim);
                void modify_dim(std::initializer_list<unsigned long> new_dim);
            private:
                void diagonal_core(Matrix<MType>& m, ul_pos mc_ul, const Matrix<MType>& fill_with, ul_pos fwc_ul, unsigned long h, unsigned long w);
                void special_jacobi_core(Matrix<MType>& m, ul_pos mc_ul, const Matrix<MType>& sjb, ul_pos fwc_ul, unsigned long jk);
                void add_padding_core(Matrix<MType>& m, Matrix<MType>& result, kernel_shape padding, ul_pos mc_ul, ul_pos fwc_ul);
                void sub_padding_core(Matrix<MType>& m, Matrix<MType>& result, kernel_shape padding, ul_pos mc_ul, ul_pos fwc_ul);
                void img2col_core(Matrix<MType>& m, Matrix<MType>& fw, ul_pos mc_ul, ul_pos fwc_ul, kernel_shape kernel_size, unsigned long stride, unsigned long output_h, unsigned long output_w);
                void col2img_core(Matrix<MType>& m, Matrix<MType> fw, ul_pos mc_ul, ul_pos fwc_ul, kernel_shape kernel_size, unsigned long stride, unsigned long output_h, unsigned long output_w);
                m_dim dim;
        };

        template<typename MType>
        void MakeMatrix<MType>::modify_dim(m_dim new_dim) {
            dim = new_dim;
        }

        template<typename MType>
        void MakeMatrix<MType>::modify_dim(std::initializer_list<unsigned long> new_dim) {
            modify_dim(m_dim {new_dim});
        }

        /* init weight */
        template<typename MType>
        void MakeMatrix<MType>::gaussian(Matrix<MType>& m) {
            std::random_device gauss_random {};
            std::mt19937 gauss_gen {gauss_random()};
            std::normal_distribution<MType> gauss_dist {0, 0.2};
            m.resize(dim, 0);
            mdata_ptr mdata = m.get_mdata();
            for (size_t i = 0; i < mdata->size(); i++) {
                mdata->at(i) = MType(gauss_dist(gauss_gen));
            }
        }

        // TODO fan_in and fan_out
        template<typename MType>
        void MakeMatrix<MType>::xavier(Matrix<MType> &m, double gain) {
            unsigned long fan_in;
            unsigned long fan_out;
            const auto std = gain * std::sqrt(6.0 / (fan_in + fan_out));
            this->uniform(m); // gaussian
        }

        // TODO
        template<typename MType>
        void MakeMatrix<MType>::kaiming(Matrix<MType> &m, double gain) {

        }

        /* init matrix */
        template<typename MType>
        void MakeMatrix<MType>::diagonal(Matrix<MType>& m, MType fill_with) {
            assert(dim[2] == dim[3]);
            m.resize(dim, 0);
            auto mdata = m.get_mdata();
            unsigned long nc = dim[0] * dim[1];
            unsigned long hw = dim[2] * dim[3];
            for (unsigned long nci = 0; nci < nc; nci++) {
                for (unsigned long hwi = 0; hwi < hw; hwi++) {
                    mdata->at(nci * hw + hwi * dim[2] + hwi) = fill_with;
                }
            }
        }

        template<typename MType>
        void MakeMatrix<MType>::diagonal(Matrix<MType>& m, const Matrix<MType>& fill_with) {
            m_dim fw_dim = fill_with.get_dim();
            assert(dim[2] % fw_dim[2] == 0);
            assert(dim[3] % fw_dim[3] == 0);
            unsigned long nh = dim[2] / fw_dim[2];
            unsigned long nw = dim[3] / fw_dim[3];
            assert(nh == nw);

            // (n, c, h, w)
            m.resize(dim, 0);
            std::vector<std::thread> threads;
            for (unsigned long n = 0; n < dim[0]; n++) {
                ul_pos mb_ul = m.get_batch(n);
                ul_pos fwb_ul = fill_with.get_batch(n);
                for (unsigned long c = 0; c < dim[1]; c++) {
                    ul_pos mc_ul = m.get_channal(c, mb_ul);
                    ul_pos fwc_ul = fill_with.get_channal(c, fwb_ul);
                    for (unsigned long i = 0; i < nh; i++) {
                        unsigned long h = i * fw_dim[2];
                        unsigned long w = i * fw_dim[3];
                        threads.emplace_back(&MakeMatrix<MType>::diagonal_core, this, std::ref(m), mc_ul, std::ref(fill_with), fwc_ul, h, w);
                    }
                }
            }
            for (auto& t : threads) {
                t.join();
            }
        }

        template<typename MType>
        void MakeMatrix<MType>::diagonal_core(Matrix<MType>& m, ul_pos mc_ul, const Matrix<MType>& fill_with, ul_pos fwc_ul, unsigned long h, unsigned long w) {
            m_dim fw_dim = fill_with.get_dim();
            m_dim dims = m.get_dim();
            auto md_ptr = m.get_mdata();

            const mdata_ptr fwd_ptr = fill_with.get_data();
            assert(h + fw_dim[2] <= dims[2]);
            assert(w + fw_dim[3] <= dims[3]);
            for (unsigned long ih = 0; ih < fw_dim[2]; ih++) {
                for (unsigned long iw = 0; iw < fw_dim[3]; iw++) {
                    md_ptr->at(
                        mc_ul.first + 
                        (ih + h) * dim[3] 
                        + w + iw
                    ) = fwd_ptr->at(
                        fwc_ul.first + 
                        ih * fw_dim[3] + 
                        iw
                    );
                }
            }
        }

        template<typename MType>
        void MakeMatrix<MType>::special_jacobi(Matrix<MType>& m, const Matrix<MType>& sj_fw, unsigned long jk) {
            m_dim dims = m.get_dim();
            m_dim sj_dims = sj_fw.get_dim();
            assert(dims[0] == sj_dims[0]);
            assert(dims[1] == sj_dims[1]);
            m.resize(dims, 0);

            std::vector<std::thread> threads;
            for (unsigned long n = 0; n < dims[0]; n++) {
                ul_pos mb_ul = m.get_batch(n);
                ul_pos fwb_ul = sj_fw.get_batch(n);
                for (unsigned long c = 0; c < dims[1]; c++) {
                    ul_pos mc_ul = m.get_channal(c, mb_ul);
                    ul_pos fwc_ul = sj_fw.get_channal(c, fwb_ul);
                    threads.emplace_back(&MakeMatrix<MType>::special_jacobi_core, this, std::ref(m), mc_ul, std::ref(sj_fw), fwc_ul, jk);
                }
            }
            for (auto& t : threads) {
                t.join();
            }
        }

        template<typename MType>
        void MakeMatrix<MType>::special_jacobi_core(Matrix<MType>& m, ul_pos mc_ul, const Matrix<MType>& sjfw, ul_pos fwc_ul, unsigned long jk) {
            m_dim dims = m.get_dim();
            m_dim sjfw_dims = sjfw.get_dim();
            mdata_ptr md_ptr = m.get_mdata();
            const mdata_ptr sjfw_ptr = sjfw.get_data();

            unsigned long offsetw;
            unsigned long fh;

            for (unsigned long h = 0; h < dims[2]; h++) {
                offsetw = h % jk;
                fh = h / jk;
                for (unsigned long fw = 0; fw < (dims[3] / jk); fw++) {
                    md_ptr->at(
                        mc_ul.first + 
                        h * dims[3] + 
                        fw * jk + 
                        offsetw
                    ) = sjfw_ptr->at(
                        fwc_ul.first + 
                        fh * sjfw_dims[3] + 
                        fw
                    );
                }
            }
        }

        template<typename MType>
        void MakeMatrix<MType>::zeros(Matrix<MType> &m) {
            mdata_ptr md_ptr = m.get_mdata();
            for (size_t i = 0; i < md_ptr->size(); i++) {
                md_ptr->at(i) = 0;
            }
            m.resize(dim, 0);
        }

        template<typename MType>
        void MakeMatrix<MType>::ones(Matrix<MType> &m) {
            mdata_ptr md_prt = m.get_mdata();
            for (size_t i = 0; i < md_prt->size(); i++) {
                md_prt->at(i) = 1;
            }
            m.resize(dim, 1);
        }

        template<typename MType>
        void MakeMatrix<MType>::identify(Matrix<MType> &m) {
            diagonal(m, 1);
        }

        template<typename MType>
        void MakeMatrix<MType>::add_padding(Matrix<MType>& m, Matrix<MType>& result, unsigned long padding, MType fill_with) {
            add_padding(m, result, kernel_shape {padding, padding}, fill_with);
        }

        template<typename MType>
        void MakeMatrix<MType>::add_padding(Matrix<MType>& m, Matrix<MType>& result, kernel_shape padding, MType fill_with) {
            // assuming padding -> (h, w)
            if (padding[0] == 0 || padding[1] == 0) {
                result = m;
                return ;
            }

            m_dim dims = m.get_dim();
            dims[2] += (padding[0] != 0) ? padding[0] * 2 : 0;
            dims[3] += (padding[1] != 0) ? padding[1] * 2 : 0;
            result.resize(dims, fill_with);
            std::vector<std::thread> threads;
            for (unsigned long n = 0; n < dims[0]; n++) {
                ul_pos mb_ul = m.get_batch(n);
                ul_pos rb_ul = result.get_batch(n);
                for (unsigned long c = 0; c < dims[1]; c++) {
                    ul_pos mc_ul = m.get_channal(c, mc_ul);
                    ul_pos rc_ul = result.get_channal(c, rb_ul);
                    threads.emplace_back(&MakeMatrix<MType>::add_padding_core, this, std::ref(m), std::ref(result), padding, mc_ul, rc_ul);
                }
            }
            for (auto& t : threads) {
                t.join();
            }
        }

        template<typename MType>
        void MakeMatrix<MType>::add_padding(Matrix<MType>& m, Matrix<MType>& result, std::initializer_list<unsigned long> kernel_size, MType fill_with) {
            add_padding(m, result, kernel_size, fill_with);
        }

        template<typename MType>
        void MakeMatrix<MType>::add_padding_core(Matrix<MType>& m, Matrix<MType>& result, kernel_shape padding, ul_pos mc_ul, ul_pos fwc_ul) {
            m_dim dims = m.get_dim();
            m_dim rdims = result.get_dim();
            mdata_ptr md_ptr = m.get_mdata();
            mdata_ptr rd_prt = result.get_mdata();

            for (unsigned long ih = 0; ih < dims[2]; ih++) {
                for (unsigned long iw = 0; iw < dims[3]; iw++) {
                    rd_prt->at(
                        fwc_ul.first + 
                        (padding[0] + ih) * rdims[3] + 
                        padding[1] + iw
                    ) = md_ptr->at(
                        mc_ul.first + 
                        ih * dims[3] + 
                        iw
                    );
                }
            }
        }

        template<typename MType>
        void MakeMatrix<MType>::sub_padding(Matrix<MType>& m, Matrix<MType>& result, unsigned long padding) {
            sub_padding(m, result, kernel_shape {padding, padding});
        }

        template<typename MType>
        void MakeMatrix<MType>::sub_padding(Matrix<MType>& m, Matrix<MType>& result, kernel_shape padding) {
            if (padding[0] == 0 || padding[1] == 0) {
                result = m;
                return ;
            }

            m_dim dims = m.get_dim();
            dims[2] -= (padding[0] != 0) ? dims[2] * 2 : 0;
            dims[3] -= (padding[1] != 0) ? dims[3] * 2 : 0;
            assert(dims[2] > 0 && dims[3] > 0);
            std::vector<std::thread> threads;
            for (unsigned long n = 0; n < dims[0]; n++) {
                ul_pos mb_ul = m.get_batch(n);
                ul_pos rb_ul = result.get_batch(n);
                for (unsigned long c = 0; c < dims[1]; c++) {
                    ul_pos mc_ul = m.get_channal(c, mb_ul);
                    ul_pos rc_ul = m.get_channal(c, rb_ul);
                    threads.emplace_back(&MakeMatrix<MType>::sub_padding_core, this, std::ref(m), std::ref(result), padding, mc_ul, rc_ul);
                }
            }
            for (auto& t : threads) {
                t.join();
            }
        }

        template<typename MType>
        void MakeMatrix<MType>::sub_padding(Matrix<MType>& m, Matrix<MType>& result, std::initializer_list<unsigned long> padding) {
            sub_padding(m, result, kernel_shape {padding});
        }

        template<typename MType>
        void MakeMatrix<MType>::sub_padding_core(Matrix<MType>& m, Matrix<MType>& result, kernel_shape padding, ul_pos mc_ul, ul_pos fwc_ul) {
            m_dim dims = m.get_dim();
            m_dim rdims = result.get_dim();
            mdata_ptr md_ptr = m.get_mdata();
            mdata_ptr rd_ptr = result.get_mdata();

            unsigned long sh = padding[0];
            unsigned long sw = padding[1];
            for (unsigned long ih = 0; ih < rdims[2]; ih++) {
                for (unsigned long iw = 0; iw < rdims[3]; iw++) {
                    rd_ptr->at(
                        fwc_ul.first + 
                        ih * rdims[3] + 
                        iw
                    ) = md_ptr->at(
                        mc_ul.first + 
                        (sh + ih) * dims[3] + 
                        sw + iw
                    );
                }
            }
        }

        template<typename MType>
        void MakeMatrix<MType>::img2col(Matrix<MType>& m, Matrix<MType>& result, unsigned long kernel_size, unsigned long stride) {
            img2col(m, result, kernel_shape {kernel_size, kernel_size}, stride);
        }

        template<typename MType>
        void MakeMatrix<MType>::img2col(Matrix<MType>& m, Matrix<MType>& result, kernel_shape kernel_size, unsigned long stride) {
            m_dim dims = m.get_dim();
            m_dim rdims = result.get_dim();
            unsigned outputh = std::floor((dims[2] - kernel_size[0]) / stride + 1);
            unsigned outputw = std::floor((dims[3] - kernel_size[1]) / stride + 1);
            rdims[2] = kernel_size[0] * kernel_size[1];
            rdims[3] = outputw * outputh;
            result.resize(rdims, 0);

            std::vector<std::thread> threads;
            for (unsigned long n = 0; n < dims[0]; n++) {
                ul_pos mb_ul = m.get_batch(n);
                ul_pos fwb_ul = result.get_batch(n);
                for (unsigned long c = 0; c < dims[1]; c++) {
                    ul_pos mc_ul = m.get_channal(c, mb_ul);
                    ul_pos fwc_ul = result.get_channal(c, fwb_ul);
                    threads.emplace_back(&MakeMatrix<MType>::img2col_core, this, std::ref(m), std::ref(result), mc_ul, fwc_ul, kernel_size, stride, outputh, outputw);
                }
            }
            for (auto& t : threads) {
                t.join();
            }
        }

        template<typename MType>
        void MakeMatrix<MType>::img2col(Matrix<MType>& m, Matrix<MType>& result, std::initializer_list<unsigned long> kernel_size, unsigned long stride) {
            img2col(m, result, kernel_shape {kernel_size}, stride);
        }

        template<typename MType>
        void MakeMatrix<MType>::img2col_core(Matrix<MType>& m, Matrix<MType>& result, ul_pos mc_ul, ul_pos fwc_ul, kernel_shape kernel_size, unsigned long stride, unsigned long outputh, unsigned long outputw) {
            mdata_ptr md_ptr = m.get_mdata();
            mdata_ptr rd_ptr = result.get_mdata();
            m_dim dims = m.get_dim();

            for (unsigned long ih = 0; ih < outputh; ih++) {
                for (unsigned long iw = 0; iw < outputw; iw++) {
                    for (unsigned long ikh = 0; ikh < kernel_size[0]; ikh++) {
                        for (unsigned long ikw = 0; ikw < kernel_size[1]; ikw++) {
                            rd_ptr->at(
                                fwc_ul.first + 
                                (ikh * kernel_size[1] + ikw) * (outputh * outputw) + 
                                ih * outputh +
                                iw
                            ) = md_ptr->at(
                                mc_ul.first + 
                                (ih * stride + ikh) * dims[3] + 
                                iw * stride + 
                                ikw
                            );
                        }
                    }
                }
            }
        }

        template<typename MType>
        void MakeMatrix<MType>::col2img(Matrix<MType>& m, Matrix<MType>& result, unsigned long kernel_size, unsigned long stride, m_dim fw) {
            col2img(m, result, kernel_shape {kernel_size, kernel_size}, stride, fw);
        }

        template<typename MType>
        void MakeMatrix<MType>::col2img(Matrix<MType>& m, Matrix<MType>& result, kernel_shape kernel_size, unsigned long stride, m_dim fw) {
            m_dim dims = m.get_dim();
            unsigned long outputh = std::floor((fw[2] - kernel_size[0]) / stride + 1);
            unsigned long outputw = std::floor((fw[3] - kernel_size[1]) / stride + 1);
            assert(outputh * outputw == dims[3]);
            assert(dims[0] == fw[0] && dims[1] == fw[1]);
            result.resize(fw, 0);

            std::vector<std::thread> threads;
            for (unsigned long n = 0; n < dims[0]; n++) {
                ul_pos mb_ul = m.get_batch(n);
                ul_pos fwb_ul = result.get_batch(n);
                for (unsigned long c = 0; c < dims[1]; c++) {
                    ul_pos mc_ul = m.get_channal(c, mb_ul);
                    ul_pos fwc_ul = m.get_channal(c, fwc_ul);
                    threads.emplace_back(&Matrix<MType>::col2img, this, std::ref(m), std::ref(result), mc_ul, fwc_ul, kernel_size, stride, outputh, outputw);
                }
            }
            for (auto& t : threads) {
                t.join();
            }
        }

        template<typename MType>
        void MakeMatrix<MType>::col2img(Matrix<MType>& m, Matrix<MType>& result, std::initializer_list<unsigned long> kernel_size, unsigned long stride, m_dim fw) {
            col2img(m, result, kernel_size, stride, fw);
        }

        template<typename MType>
        void MakeMatrix<MType>::col2img_core(Matrix<MType>& m, Matrix<MType> fw, ul_pos mc_ul, ul_pos fwc_ul, kernel_shape kernel_size, unsigned long stride, unsigned long output_h, unsigned long output_w) {
            m_dim dims = m.get_dim();
            m_dim fwdims = fw.get_dim();
            mdata_ptr md_ptr = m.get_mdata();
            mdata_ptr fwd_ptr = fw.get_mdata();

            for (unsigned long ih = 0; ih < output_h; ih++) {
                for (unsigned long iw = 0; iw < output_w; iw++) {
                    for (unsigned long ikh = 0; ikh < kernel_size[0]; ikh++) {
                        for (unsigned long ikw = 0; ikw < kernel_size[1]; ikw++) {
                            fwd_ptr->at(
                                fwc_ul.first + 
                                (ih * stride + ikh) * fwdims[2] + 
                                iw * stride + 
                                ikw
                            ) = md_ptr(
                                mc_ul.first + 
                                (ikh * kernel_size[2] + ikw) * dims[3] + 
                                ih * output_h + 
                                iw
                            );
                        }
                    }
                }
            }
        }

    }
}