#include "N3LDG.h"
#include <memory>

constexpr int dim = 2000;

struct ModelParams : public N3LDGSerializable, public TunableCombination<BaseParam>
#if USE_GPU
, public TransferableComponents
#endif
{
    vector<UniParams*> params;
    vector<UniParams*> vector_params;
    vector<UniParams*> small_vector_params;
    UniParams output_param;
    UniParams output_param2;

    ModelParams() : output_param("output_param"), output_param2("output_param2") {
        for (int i = 0; i < 10; ++i) {
            UniParams* param = new UniParams(to_string(i));
            param->init(dim, dim);
            params.push_back(param);

            UniParams *vector_param = new UniParams("vector" + to_string(i));
            vector_param->init(dim, dim);
            vector_params.push_back(vector_param);

            UniParams *small_vector_param = new UniParams("small-vector" + to_string(i));
            small_vector_param->init(i + 10, i + 10);
            small_vector_params.push_back(small_vector_param);
        }
        output_param.init(3, 10 * dim);
        output_param2.init(3, dim);
    } 

    Json::Value toJson() const override {
        Json::Value json;
        return json;
    }

    void fromJson(const Json::Value &json) override {
    }

    template<typename T>
    vector<T *> children() {
        vector<T *> ptrs = {&output_param, &output_param2};
        for (auto x : params) {
            ptrs.push_back(x);
        }
        for (auto x : vector_params) {
            ptrs.push_back(x);
        }
        for (auto x : small_vector_params) {
            ptrs.push_back(x);
        }
        return ptrs;
    }

#if USE_GPU
    vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        return children<n3ldg_cuda::Transferable>();
    }
#endif

protected:
    vector<Tunable<BaseParam>*> tunableComponents() override {
        return children<Tunable<BaseParam>>();
    }
};

int main() {
#if USE_GPU
    n3ldg_cuda::InitCuda(0, 0);
#endif
    ModelParams params;
#if !USE_GPU
    CheckGrad grad_checker;
    grad_checker.init(params.tunableParams());
#endif
    ModelUpdate optimizer;
    optimizer.setParams(params.tunableParams());
    optimizer._alpha = 1e-5;
    using namespace n3ldg_plus;

    auto f = [&](int answer) {
        shared_ptr<Graph> graph(new Graph);
        Node *bucket = n3ldg_plus::bucket(*graph, dim, 1);
        vector<Node *> ys;
        for (int i = 0; i < 10; ++i) {
            vector<Node *> nodes;
            for (int j = 0; j < 10 - i; ++j) {
                Node *x = n3ldg_plus::linear(*graph, *params.params.at(j), *bucket);
                nodes.push_back(x);
            }
            MatrixNode * y = n3ldg_plus::concatToMatrix(*graph, nodes);
            Node *v = n3ldg_plus::linear(*graph, *params.vector_params.at(i),
                    *n3ldg_plus::bucket(*graph, dim, 1));
            y = n3ldg_plus::matrixPointwiseMultiply(*graph, *y, *v);
            Node *col_sum = n3ldg_plus::matrixColSum(*graph, *y);
            Node *z;
            if (i > 0) {
                z = n3ldg_plus::concat(*graph, {y, n3ldg_plus::bucket(*graph,
                            i * dim - col_sum->getDim(), 0), col_sum});
            } else {
                z = y;
            }
            ys.push_back(z);
        }
        Node *output = n3ldg_plus::averagePool(*graph, ys);
        output = n3ldg_plus::linear(*graph, params.output_param, *output);
        output = n3ldg_plus::softmax(*graph, *output);
        vector<Node *> v = {output};
        graph->compute();
//        output->getVal().print();
//        cout << "answer:" << answer << endl;
        dtype loss = crossEntropyLoss(v, {answer}, 1);
//        cout << "loss:" << loss << endl;
        return make_pair(graph, loss);
    };

    for (int i = 0; i < 10; ++i) {
        int answer = i % 3;
        auto p = f(answer);
        p.first->backward();

#if !USE_GPU
        auto loss_function = [&](const int &p) -> dtype {
            return f(answer).second;
        };
        vector<int> nulls = {1};
        grad_checker.check<int>(loss_function, nulls, "");
#endif

        optimizer.updateAdam();
    }

    auto g = [&](int answer) {
        shared_ptr<Graph> graph(new Graph);
        vector<Node *> to_pool_nodes;
        for (int i = 10; i < 20; ++i) {
            vector<Node *> buckets;
            for (int j = 0; j < i; ++j) {
                Node *bucket_node = bucket(*graph, dim, 1);
                bucket_node = linear(*graph, *params.params.at(j % 10), *bucket_node);
                buckets.push_back(bucket_node);
            }
            MatrixNode *matrix = concatToMatrix(*graph, buckets);
            Node *vector_node = bucket(*graph, i, 1);
            vector_node = linear(*graph, *params.small_vector_params.at(i % 10), *vector_node);

            vector_node = matrixAndVectorMulti(*graph, *matrix, *vector_node);
            to_pool_nodes.push_back(vector_node);
        }
        Node *output = n3ldg_plus::averagePool(*graph, to_pool_nodes);
        output = n3ldg_plus::linear(*graph, params.output_param2, *output);
        output = n3ldg_plus::softmax(*graph, *output);
        vector<Node *> v = {output};
        graph->compute();
        //        output->getVal().print();
        //        cout << "answer:" << answer << endl;
        dtype loss = crossEntropyLoss(v, {answer}, 1);
        //        cout << "loss:" << loss << endl;
        return make_pair(graph, loss);
    };

    for (int i = 0; i < 10; ++i) {
        int answer = i % 3;
        auto p = g(answer);
        p.first->backward();

#if !USE_GPU
        auto loss_function = [&](const int &p) -> dtype {
            return g(answer).second;
        };
        vector<int> nulls = {1};
        grad_checker.check<int>(loss_function, nulls, "");
#endif

        optimizer.updateAdam();
    }

    return 0;
}
