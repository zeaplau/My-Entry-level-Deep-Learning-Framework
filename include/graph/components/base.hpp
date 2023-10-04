#pragma once
#include "../node/common/base.hpp"
#include <initializer_list>
#include <memory>

namespace dddl {
    namespace components {
        template<typename MType>
        class Component {
            public:
                // alias
                using node_ptr = std::shared_ptr<graph::Node<MType>>;
                using node_ptrs = std::shared_ptr<std::vector<node_ptr>>; // for channal

                // construct & desctruct
                Component() = default;
                virtual ~Component();

                // function
                virtual void forward() = 0;
                virtual void backward(node_ptr end) = 0;
                virtual void update(MType lr) {};
                virtual void construct(node_ptrs input_nodes) = 0;
                virtual node_ptrs operator()(std::initializer_list<node_ptr> input_nodes) = 0;
                virtual node_ptrs operator()(node_ptrs input_nodes) = 0;
                virtual node_ptrs get_output_nodes();
                virtual Matrix<MType> get_data() = 0;
                virtual void clear_jacobi() = 0;
            protected:
                node_ptrs ins;
                node_ptrs outs;
                std::string layer_name;
                bool complete_construct {false};
        };

        template<typename MType>
        Component<MType>::~Component<MType>() { }

        template<typename MType>
        typename Component<MType>::node_ptrs Component<MType>::get_output_nodes() {
            return outs;
        }
    }
}