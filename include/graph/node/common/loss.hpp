#pragma once
#include "base.hpp"
#include <memory>
#include <vector>
#include <stdexcept>

namespace dddl {
    namespace graph {
        template<typename MType>
        class LossNode : public Node<MType> {
            public:
                using nodes = std::shared_ptr<std::vector<std::shared_ptr<Node<MType>>>>;
                using node_ptr = std::shared_ptr<Node<MType>>;
                using md_ptr = std::shared_ptr<std::vector<MType>>;
                using m_dim = std::vector<unsigned long>;
                using Node<MType>::Node;

                nodes get_children() override;
                node_ptr get_children(size_t child_id) override;
                size_t get_children_len() override;
                void add_children(node_ptr child) override;
                void backward(node_ptr output_node) override = 0;
        };

        template<typename MType>
        typename LossNode<MType>::nodes LossNode<MType>::get_children() {
            throw std::runtime_error("LossNode is not allowed to get children");
        }

        template<typename MType>
        typename LossNode<MType>::node_ptr LossNode<MType>::get_children(size_t child_id) {
            throw std::runtime_error("LossNode is not allowed to get children");
        }

        template<typename MType>
        size_t LossNode<MType>::get_children_len() {
            return 0;
        }

        template<typename MType>
        void LossNode<MType>::add_children(node_ptr child) {
            throw std::runtime_error("LossNode is not allowed to add child");
        }
    }
}
