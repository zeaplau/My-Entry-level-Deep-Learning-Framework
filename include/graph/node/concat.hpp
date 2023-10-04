#pragma once
#include "common/base.hpp"
#include <memory>
#include <utility>
#include <vector>

namespace dddl {
    namespace graph {
        template<typename MType>
        class ConcatNode : public Node<MType> {
            public:
                // alias
                using ul_pos = const std::pair<unsigned long, unsigned long>;
                using node_ptr = std::shared_ptr<Node<MType>>;
                using kernel_shape = std::vector<unsigned long>;
                using m_dim = std::vector<unsigned long>;
                using md_ptr = std::shared_ptr<std::vector<MType>>;
                using nodes = std::shared_ptr<std::vector<std::shared_ptr<Node<MType>>>>;

                // construct & destruct
                ConcatNode(std::string name, m_dim dims, unsigned long concat_dims) : Node<MType> {name, dims} {};
                ConcatNode(std::string name, const Matrix<MType> &m, unsigned long concat_dim) : Node<MType> {name, m}, concat_dim_(concat_dim) {};
                ConcatNode(std::string name, md_ptr data, m_dim dims, unsigned long concat_dim) : Node<MType> {name, data, dims}, concat_dim_(concat_dim) {};
                ConcatNode(std::string name, md_ptr data, m_dim dims, node_ptr parent, unsigned long concat_dim) : Node<MType> {name, data, dims, parent}, concat_dim_(concat_dim) {};
                ConcatNode(std::string name, md_ptr data, m_dim dims, nodes parents, unsigned long concat_dim) : Node<MType> {name, data, dims, parents}, concat_dim_(concat_dim) {};
                void compute_forward() override;
                void compute_jacobi(Matrix<MType>& m, node_ptr parent_node) override;
                void backward(node_ptr output_node) override;
            protected:
                matrix_tools::MakeMatrix<MType> mm;
                unsigned long concat_dim_;
        };

        template<typename MType>
        void ConcatNode<MType>::compute_forward() {
            size_t parents_len = Node<MType>::get_parents_len();
            Matrix<MType>::data = Node<MType>::get_parent(0)->get_data();
            for (size_t i = 1; i < parents_len; i++) {
                Matrix<MType>::data.concat(Node<MType>::get_parent(i)->get_dim(), concat_dim_);
            }
        }

        template<typename MType>
        void ConcatNode<MType>::compute_jacobi(Matrix<MType>& m, node_ptr parent_node) {
            size_t parents_len = Node<MType>::get_parents_len();
            int pi = 0;
            while (parent_node != Node<MType>::get_parent(pi) && pi != parents_len) {
                ++pi;
            }
            assert(pi < parents_len);
            m = Node<MType>::get_parent(pi)->get_data().slice(1, {pi, pi + 1});
        }

        template<typename MType>
        void ConcatNode<MType>::backward(node_ptr output_node) {

        }

    }
}