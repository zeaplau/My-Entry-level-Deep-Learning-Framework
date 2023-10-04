#include "../../../math/matrix.hpp"
#include "../../../math/tools.hpp"
#include <initializer_list>
#include <memory>
#include <string>
#include <type_traits>
#include <vcruntime.h>
#include <vector>

namespace dddl {
    namespace graph {
        template<typename MType>
        class Node : public std::enable_shared_from_this<Node<MType>> {
            public:
                // alias
                using nodes = std::shared_ptr<std::vector<std::shared_ptr<Node>>>;
                using node_ptr = std::shared_ptr<Node>;
                using md_ptr = std::shared_ptr<std::vector<MType>>;
                using mdata = std::shared_ptr<Matrix<MType>>;
                using m_dim = std::vector<unsigned long>;

                // construct & descruct
                Node(std::string name, m_dim dims);
                Node(std::string name, const Matrix<MType>& m);
                Node(std::string name, md_ptr ptr, m_dim dims);
                Node(std::string name, md_ptr ptr, m_dim dims, node_ptr parent);
                Node(std::string name, md_ptr ptr, m_dim dims, nodes parents);
                virtual ~Node();

                // children & parent nodes
                virtual nodes get_children();
                virtual nodes get_parents();
                virtual node_ptr get_children(size_t children_id);
                virtual node_ptr get_parent(size_t parent_id);
                virtual void forward();
                virtual void backward(node_ptr output_node);
                virtual void compute_forward() {}; // TODO
                virtual void compute_jacobi(Matrix<MType>& m, node_ptr parent_node) {}; // TODO
                virtual void no_grad();
                virtual void ask_grad();
                virtual void add_parent(node_ptr parent);
                virtual void add_children(node_ptr children);
                virtual size_t get_parents_len();
                virtual void set_data(const Matrix<MType>& m);
                virtual void clear_jacobi();
                virtual void update(MType lr) {};
                virtual Matrix<MType> get_data();
                virtual Matrix<MType> get_jacobi();
                virtual md_ptr get_mdata();
                virtual md_ptr get_mjacobi();
                virtual m_dim get_data_dim();
                virtual m_dim get_jacobi_dim();

                // view
                virtual void view_data(m_dim shape);
                virtual void view_data(unsigned long n, unsigned long c, unsigned long h, unsigned long w);
                virtual void view_data(std::initializer_list<unsigned long> shape);
                virtual void view_jacobi(m_dim shape);
                virtual void view_jacobi(unsigned long n, unsigned long c, unsigned long h, unsigned long w);
                virtual void view_jacobi(std::initializer_list<unsigned long> shape);

                // other
                virtual void init_data(std::string mode) {};
                bool is_jacobi_exists();
            protected:
                nodes parents {std::make_shared<std::vector<std::shared_ptr<Node>>>()};
                nodes children {std::make_shared<std::vector<std::shared_ptr<Node>>>()};
                std::string name;
                Matrix<MType> data {};
                Matrix<MType> jacobi {};
                bool wait_backward {false};
                bool require_grad {true};
        };

        // construct & destruct
        template<typename MType>
        Node<MType>::Node(std::string name, m_dim dims) {
            this->name = name;
            this->data = Matrix<MType>(dims, MType(0));
            m_dim empty_jacobi {1, 1, 1, 1};
            jacobi = Matrix<MType>(empty_jacobi, MType(0));
        }

        template<typename MType>
        Node<MType>::Node(std::string name, const Matrix<MType>& m) {
            this->name = name;
            this->data = m;
            m_dim empty_jacobi {1, 1, 1, 1};
            jacobi = Matrix<MType>(empty_jacobi, MType(0));
        }

        template<typename MType>
        Node<MType>::Node(std::string name, md_ptr ptr, m_dim dims) {
            this->name = name;
            this->data = Matrix<MType>(ptr, dims);
            m_dim empty_jacobi {1, 1, 1, 1};
            jacobi = Matrix<MType>(empty_jacobi, MType(0));
        }

        template<typename MType>
        Node<MType>::Node(std::string name, md_ptr ptr, m_dim dims, node_ptr parent) {
            this->name = name;
            this->data = Matrix<MType>(ptr, dims);
            m_dim empty_jacobi {1, 1, 1, 1};
            jacobi = Matrix<MType>(empty_jacobi, MType(0));
            add_parent(parent);
            parent->add_children(std::make_shared<node_ptr>(this));
        }

        template<typename MType>
        Node<MType>::Node(std::string name, md_ptr ptr, m_dim dims, nodes parents) {
            this->parents.reset();
            this->parents = parents;
            this->name = name;
            this->data = Matrix<MType>(ptr, dims);
            m_dim empty_jacobi {1, 1, 1, 1};
            jacobi = Matrix<MType>(empty_jacobi, MType(0));
            for (auto& pi : parents) {
                pi->add_children(std::make_shared<node_ptr>(this));
            }
        }

        template<typename MType>
        Node<MType>::~Node() {
            for (auto& child : children) {
                child.reset();
            }
            children.reset();
        }

        // children & parent nodes
        template<typename MType>
        typename Node<MType>::nodes Node<MType>::get_children() {
            return children;
        }

        template<typename MType>
        typename Node<MType>::node_ptr Node<MType>::get_children(size_t children_id) {
            if (children == nullptr) {
                return nullptr;
            }
            return children->at(children_id);
        }

        template<typename MType>
        typename Node<MType>::nodes Node<MType>::get_parents() {
            return parents;
        }

        template<typename MType>
        typename Node<MType>::node_ptr Node<MType>::get_parent(size_t parent_id) {
            if (parents == nullptr) {
                return nullptr;
            }
            return parents->at(parent_id);
        }

        template<typename MType>
        size_t Node<MType>::get_parents_len() {
            if (parents == nullptr) {
                return 0;
            }
            return parents->size();
        }


        template<typename MType>
        void Node<MType>::forward() {
            for (size_t pi = 0; pi < parents->size(); pi++) {
                if (!parents->at(pi)->wait_backward) {
                    parents->at(pi)->forward();
                }
            }
            compute_forward();
            wait_backward = true;
        }

        template<typename MType>
        void Node<MType>::backward(node_ptr output_node) {
            if (!require_grad) {
                // accumulate grad
                for (size_t ci = 0; ci < children->size(); ci++) {
                    jacobi += children->at(ci)->jacobi;
                }
                return ;
            }
            // this node is output node
            if (std::enable_shared_from_this<Node<MType>>::shared_from_this() == output_node) {
                m_dim j_dims = Node<MType>::data.get_dim();
                unsigned long j_shape = j_dims[2] * j_dims[3];
                j_dims[2] = j_shape;
                j_dims[3] = j_shape;
                matrix_tools::MakeMatrix<MType> mm = j_dims;
                mm.identify(Node<MType>::jacobi);
            }
            // backward children
            m_dim tmp_dims {0, 0, 0, 0};
            m_dim child_dims {output_node->data.get_dim()};
            m_dim this_dims {data.get_dim()};
            Matrix<MType> tmp_m = Matrix<MType>(tmp_dims, MType(0));
            // (n, c, ch * cw, th * tw), each child grad for this grad
            jacobi.resize(this_dims[0], this_dims[1], child_dims[2] * child_dims[3], this_dims[2] * this_dims[3], 0);
            for (size_t ci = 0; ci < children->size(); ci++) {
                if (children->at(ci)->wait_backward) {
                    children->at(ci)->backward(output_node);
                }
                children->at(ci)->compute_jacobi(tmp_m, std::enable_shared_from_this<Node<MType>>::share_from_this());
                m_dim child_jacobi_dims = children->at(ci)->jacobi->get_dim();
                m_dim tmp_jacobi_dims = tmp_m.get_dim();
                // What this mean?
                if (child_jacobi_dims[2] == tmp_jacobi_dims[3]) {
                    jacobi += children->at(ci)->jacobi * tmp_m;
                } else {
                    jacobi += tmp_m.mul_v(children->at(ci)->jacobi);
                }
            }
            jacobi.view(data.get_dim());
            wait_backward = false;
        }

        template<typename MType>
        void Node<MType>::no_grad() {
            require_grad = false;
        }

        template<typename MType>
        void Node<MType>::ask_grad() {
            require_grad = true;
        }

        template<typename MType>
        void Node<MType>::add_parent(node_ptr parent) {
            if (parents == nullptr) {
                parents = std::make_shared<std::vector<std::shared_ptr<Node>>>();
            }
            for (size_t pi = 0; pi < parents->size(); pi++) {
                if (parent == parents->at(pi)) {
                    return;
                }
            }
            parents->push_back(parent);
            parent->add_children(std::enable_shared_from_this<Node<MType>>::share_from_this());
        }

        template<typename MType>
        void Node<MType>::add_children(node_ptr child) {
            if (children == children) {
                children = std::make_shared<std::vector<std::shared_ptr<Node>>>();
            }
            for (size_t ci = 0; ci < children->size(); ci++) {
                if (child == children->at(ci)) {
                    return ;
                }
            }
            children->push_back(child);
            child->add_parent(std::enable_shared_from_this<Node<MType>>::share_from_this());
        }

        template<typename MType>
        void Node<MType>::set_data(const Matrix<MType> &m) {
            data = m;
        }

        template<typename MType>
        void Node<MType>::clear_jacobi() {
            jacobi.clear_data();
        }
        
        template<typename MType>
        Matrix<MType> Node<MType>::get_data() {
            return data;
        }

        template<typename MType>
        Matrix<MType> Node<MType>::get_jacobi() {
            return jacobi;
        }

        template<typename MType>
        typename Node<MType>::md_ptr Node<MType>::get_mdata() {
            return std::make_shared<Matrix<MType>>(data);
        }

        template<typename MType>
        typename Node<MType>::md_ptr Node<MType>::get_mjacobi() {
            return std::make_shared<Matrix<MType>>(jacobi);
        }

        template<typename MType>
        typename Node<MType>::m_dim Node<MType>::get_data_dim() {
            return data.get_dim();
        }

        template<typename MType>
        typename Node<MType>::m_dim Node<MType>::get_jacobi_dim() {
            return jacobi.get_dim();
        }

        // TODO
        template<typename MType>
        void Node<MType>::view_data(m_dim shape) {

        }

        template<typename MType>
        void Node<MType>::view_data(unsigned long n, unsigned long c, unsigned long h, unsigned long w) {

        }

        template<typename MType>
        void Node<MType>::view_data(std::initializer_list<unsigned long> shape) {

        }

        template<typename MType>
        void Node<MType>::view_jacobi(m_dim shape) {

        }

        template<typename MType>
        void Node<MType>::view_jacobi(unsigned long n, unsigned long c, unsigned long h, unsigned long w) {

        }

        template<typename MType>
        void Node<MType>::view_jacobi(std::initializer_list<unsigned long> shape) {

        }

        // other
        template<typename MType>
        bool Node<MType>::is_jacobi_exists() {

        }

    }
}