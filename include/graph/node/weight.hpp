#pragma once
#include <map>
#include <stdint.h>
#include "common/base.hpp"

namespace dddl {
    namespace graph {
        template <typename MType>
        class WeightNode : public Node<MType> {
            using Node<MType>::Node;
            void init_data(std::string init_method) override;
            void update(MType lr) override;
        };

        template <typename MType>
        void WeightNode<MType>::init_data(std::string init_method) {
            matrix_tools::MakeMatrix<MType> mm = Node<MType>::data.get_dim();
            std::map<std::string, void (matrix_tools::MakeMatrix<MType>::*)(Matrix<MType>&)> func_map {
                {std::string("gaussian", &matrix_tools::MakeMatrix<MType>::gaussian)}, 
            };
            auto func_map_iter = func_map.find(init_method);
            assert(func_map_iter != func_map.end());
            (mm.*func_map[init_method])(Node<MType>::data);
        }

        template <typename MType>
        void WeightNode<MType>::update(MType lr) {
            Node<MType>::data += Node<MType>::jacobi * (-1.0 * lr);
        }
    }
}