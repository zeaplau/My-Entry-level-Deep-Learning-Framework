#pragma once
#include "common/base.hpp"
#include <memory>
#include <vcruntime.h>

namespace dddl {
    namespace graph {
        template<typename MType>
        class AddNode : public Node<MType> {
            public:
                using node_ptr = std::shared_ptr<Node<MType>>;
                using Node<MType>::Node;
                void compute_forward() override;
                void compute_jacobi(Matrix<MType>& m, node_ptr parent_node) override;
        };

        template<typename MType>
        void AddNode<MType>::compute_forward() {
            size_t parents_len = Node<MType>::get_parents_len();
            // must more than 2 nodes
            assert(parents_len >= 2);
            Node<MType>::data.copy_from(Node<MType>::get_parents(0)->get_data());
            for (size_t i = 0; i < parents_len; i++) {
                Node<MType>::data += Node<MType>::get_parents(i)->get_data();
            }
        }

        template<typename MType>
        void AddNode<MType>::compute_jacobi(Matrix<MType> &m, node_ptr parent_node) {
            typename Node<MType>::m_dim j_dim = Node<MType>::data.get_dim();
            unsigned long new_j_dim = j_dim[2] * j_dim[3];
            j_dim[2] = new_j_dim;
            j_dim[3] = new_j_dim;
            matrix_tools::MakeMatrix<MType> mm {j_dim};
            mm.identify(m);
        }
    }
}