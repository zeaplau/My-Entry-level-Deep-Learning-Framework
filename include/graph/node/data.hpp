#pragma once
#include "common/base.hpp"
#include <memory>
#include <stdexcept>
#include <vcruntime.h>
#include <vector>

namespace dddl {
    namespace graph {
    template<typename MType>
    class DataNode : public Node<MType> {
        public:
            using Node<MType>::Node;
            using nodes = std::shared_ptr<std::vector<std::shared_ptr<Node<MType>>>>;
            using node_ptr = std::shared_ptr<Node<MType>>;
            using md_ptr = std::shared_ptr<std::vector<MType>>;
            using m_dim = std::vector<unsigned long>;

            DataNode(std::string name, m_dim dims, md_ptr data, node_ptr parent) {
                throw std::runtime_error("`DataNode` is not allowed to modity `parent`");
            };
            DataNode(std::string name, m_dim dims, md_ptr data, nodes parents) : Node<MType> {name, dims, parents} {
                throw std::runtime_error("`DataNode` is not allowed to modify `parent`");
            };
            
            nodes get_parents() override;
            node_ptr get_parent() override;
            size_t get_parents_len() override;

            void add_parent(node_ptr parent) override;
            void compute_forward() override {};
            void forward() override {};
            void backward(node_ptr child) override {};
            void clear_jacobi() override {};
        protected:
            bool required_grad = false;
    };

    template<typename MType>
    typename DataNode<MType>::nodes DataNode<MType>::get_parents() {
        throw std::runtime_error("`DataNode` is not allowed to modify `parent`");
    }

    template<typename MType>
    typename DataNode<MType>::node_ptr DataNode<MType>::get_parent() {
        throw std::runtime_error("`DataNode` is not allowed to modify `parent`");
    }

    template<typename MType>
    size_t DataNode<MType>::get_parents_len() {
        return 0;
    }

    template<typename MType>
    void DataNode<MType>::add_parent(node_ptr parent) {
        throw std::runtime_error("`DataNode` is not allowed to modify `parent`");
    }
    }
}