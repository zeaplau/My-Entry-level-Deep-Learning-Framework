#pragma once
#include "common/base.hpp"
#include <memory>
#include <vector>

namespace dddl {
    namespace graph {
        template <typename MType>
        class PaddingNode : public Node<MType> {
            public:
                // alias
                using ul_pos = const std::pair<unsigned long, unsigned long>;
                using node_ptr = std::shared_ptr<Node<MType>>;
                using kernel_shape = std::vector<unsigned long>;
                using m_dim = std::vector<unsigned long>;
                using md_ptr = std::shared_ptr<std::vector<MType>>;
                using nodes = std::shared_ptr<std::vector<std::shared_ptr<std::vector<MType>>>>;

                // construct & desctruct
                PaddingNode(std::string name, m_dim dims, kernel_shape padding, MType padding_init) : 
                Node<MType> {name, dims}, padding_size_(padding), padding_init_(padding_init) {};
                PaddingNode(std::string name, md_ptr data, m_dim dims, kernel_shape padding, MType padding_init) : 
                Node<MType> {name, data, dims}, padding_size_(padding), padding_init_(padding_init) {};
                PaddingNode(std::string name, const Matrix<MType>& m, kernel_shape padding, MType padding_init) : Node<MType> {name, m}, padding_size_(padding), padding_init_(padding_init) {}; 
                PaddingNode(std::string name, md_ptr data, m_dim dims, node_ptr parent, kernel_shape padding, MType padding_init) : Node<MType> {name, data, dims, parent}, padding_size_(padding), padding_init_(padding_init) {};
                PaddingNode(std::string name, md_ptr data, m_dim dims, nodes parents, kernel_shape padding, MType padding_init) : Node<MType> {name, data, dims, parents}, padding_size_(padding), padding_init_(padding_init) {};

                // function
                void compute_forward() override;
                void compute_jacobi(Matrix<MType>& m, node_ptr parent_node) override;
                void backward(node_ptr output_node) override;
            private:
                matrix_tools::MakeMatrix<MType> mm;
                kernel_shape padding_size_;
                MType padding_init_;
        };

        template <typename MType>
        void PaddingNode<MType>::compute_forward() {
            size_t parents_len = Node<MType>::get_parents_len();
            assert(parents_len == 1);
            if (padding_size_[0] == 0 && padding_size_[1] == 0) {
                Node<MType>::data = Node<MType>::get_parent(0)->get_data();
                return ;
            }
            mm.modify_dim(Node<MType>::data.get_dim());
            mm.add_padding(Node<MType>::get_parent(0), Node<MType>::data, padding_size_, padding_init_);
        }

        template <typename MType>
        void PaddingNode<MType>::compute_jacobi(Matrix<MType> &m, node_ptr parent_node) {
            assert(parent_node == Node<MType>::get_parent(0));
            mm.modify_dim(parent_node->get_data_dim());
            mm.sub_padding(Node<MType>::data, m, padding_size_);
        }

        template <typename MType>
        void PaddingNode<MType>::backward(node_ptr output_node) {
            compute_jacobi(Node<MType>::jacobi, output_node);
            Node<MType>::wait_backward = false;
        }
    }
}