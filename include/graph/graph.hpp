#pragma once
#include "./node/common/base.hpp"
#include "../math/matrix.hpp"
#include <initializer_list>
#include <memory>
#include <vector>

namespace dddl {
    namespace graph {
        template<typename MType>
        class Graph {
            public:
                using node_ptr = std::shared_ptr<Node<MType>>;
                using node_ptrs = std::vector<std::shared_ptr<Node<MType>>>;
            
                virtual ~Graph();
                Graph() {};
                virtual void forward(std::initializer_list<Matrix<MType>> input_m) = 0;
                void clear_jacobi();
            protected:
        };
    }
}