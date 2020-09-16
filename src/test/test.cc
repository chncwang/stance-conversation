#include "N3LDG.h"
#include <memory>

constexpr int dim = 2000;

struct ModelParams : public N3LDGSerializable, public TunableCombination<BaseParam>
#if USE_GPU
, public TransferableComponents
#endif
{
    vector<UniParams*> params;
    UniParams output_param;

    ModelParams() : output_param("output_param") {
        for (int i = 0; i < 10; ++i) {
            UniParams* param = new UniParams(to_string(i));
            param->init(dim, dim);
            params.push_back(param);
        }
        output_param.init(3, 10 * dim);
    } 

    Json::Value toJson() const override {
        Json::Value json;
        return json;
    }

    void fromJson(const Json::Value &json) override {
    }

#if USE_GPU
    vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        vector<n3ldg_cuda::Transferable *> ptrs = {&output_param};
        for (auto x : params) {
            ptrs.push_back(x);
        }
        return ptrs;
    }
#endif

protected:
    virtual vector<Tunable<BaseParam>*> tunableComponents() override {
        vector<Tunable<BaseParam> *> ptrs = {&output_param};
        for (auto x : params) {
            ptrs.push_back(x);
        }
        return ptrs;
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
            Node * y = n3ldg_plus::concatToMatrix(*graph, nodes);
            if (i > 0) {
                y = n3ldg_plus::concat(*graph, {y, n3ldg_plus::bucket(*graph, i * dim, 0)});
            }
            ys.push_back(y);
        }
        Node *output = n3ldg_plus::averagePool(*graph, ys);
        output = n3ldg_plus::linear(*graph, params.output_param, *output);
        output = n3ldg_plus::softmax(*graph, *output);
        vector<Node *> v = {output};
        graph->compute();
        output->getVal().print();
        cout << "answer:" << answer << endl;
        dtype loss = crossEntropyLoss(v, {answer}, 1);
        cout << "loss:" << loss << endl;
        return make_pair(graph, loss);
    };

    for (int i = 0; i < 1000; ++i) {
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

    return 0;
}
