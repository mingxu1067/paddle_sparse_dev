#include "paddle/fluid/operators/mul_sparse_op.h"
#include "paddle/fluid/operators/mul_op.h"

namespace paddle {
namespace operators {

using framework::OpKernelType;
using framework::Tensor;

class MulSparseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "MulSparse");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "MulSparse");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "MulSparse");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    int x_num_col_dims = ctx->Attrs().Get<int>("x_num_col_dims");
    int y_num_col_dims = ctx->Attrs().Get<int>("y_num_col_dims");

    VLOG(3) << "mul operator x.shape=" << x_dims << " y.shape=" << y_dims
            << " x_num_col_dims=" << x_num_col_dims
            << " y_num_col_dims=" << y_num_col_dims;

    PADDLE_ENFORCE_NE(framework::product(y_dims), 0,
                      platform::errors::PreconditionNotMet(
                          "The Input variable Y(%s) has not "
                          "been initialized. You may need to confirm "
                          "if you put exe.run(startup_program) "
                          "after optimizer.minimize function.",
                          ctx->Inputs("Y").front()));
    PADDLE_ENFORCE_GT(
        x_dims.size(), x_num_col_dims,
        platform::errors::InvalidArgument(
            "The input tensor X's dimensions of MulOp "
            "should be larger than x_num_col_dims. But received X's "
            "dimensions = %d, X's shape = [%s], x_num_col_dims = %d.",
            x_dims.size(), x_dims, x_num_col_dims));
    PADDLE_ENFORCE_GT(
        y_dims.size(), y_num_col_dims,
        platform::errors::InvalidArgument(
            "The input tensor Y's dimensions of MulOp "
            "should be larger than y_num_col_dims. But received Y's "
            "dimensions = %d, Y's shape = [%s], y_num_col_dims = %d.",
            y_dims.size(), y_dims, y_num_col_dims));

    auto x_mat_dims = framework::flatten_to_2d(x_dims, x_num_col_dims);
    auto y_mat_dims = framework::flatten_to_2d(y_dims, y_num_col_dims);

    std::vector<int64_t> output_shape_vec = ctx->Attrs().Get<std::vector<int>>("output_shape");
    if (output_shape_vec.size() > 0) {
        ctx->SetOutputDim("Out", framework::make_ddim(output_shape_vec));
    } else {
      PADDLE_ENFORCE_EQ(
          x_mat_dims[1], y_mat_dims[0],
          platform::errors::InvalidArgument(
              "After flatten the input tensor X and Y to 2-D dimensions matrix "
              "X1 and Y1, the matrix X1's width must be equal with matrix Y1's "
              "height. But received X's shape = [%s], X1's shape = [%s], X1's "
              "width = %s; Y's shape = [%s], Y1's shape = [%s], Y1's height = "
              "%s.",
              x_dims, x_mat_dims, x_mat_dims[1], y_dims, y_mat_dims,
              y_mat_dims[0]));
      std::vector<int64_t> output_dims;
      output_dims.reserve(
          static_cast<size_t>(x_num_col_dims + y_dims.size() - y_num_col_dims));

      for (int i = 0; i < x_num_col_dims; ++i) {
        output_dims.push_back(x_dims[i]);
      }

      for (int i = y_num_col_dims; i < y_dims.size(); ++i) {
        output_dims.push_back(y_dims[i]);
      }

      ctx->SetOutputDim("Out", framework::make_ddim(output_dims));
    }
    ctx->ShareLoD("X", /*->*/ "Out");
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    int customized_type_value =
        framework::OpKernelType::kDefaultCustomizedTypeValue;
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");

    return framework::OpKernelType(input_data_type, ctx.GetPlace(), layout,
                                   library, customized_type_value);
  }
};

class MulSparseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The first input tensor of mul op.");
    AddInput("Y", "(Tensor), The second input tensor of mul op.");
    AddOutput("Out", "(Tensor), The output tensor of mul op.");
    AddAttr<int>(
        "x_num_col_dims",
        R"DOC((int, default 1), The mul_op can take tensors with more than two
              dimensions as its inputs. If the input $X$ is a tensor with more
              than two dimensions, $X$ will be flattened into a two-dimensional
              matrix first. The flattening rule is: the first `num_col_dims`
              will be flattened to form the first dimension of the final matrix
              (the height of the matrix), and the rest `rank(X) - num_col_dims`
              dimensions are flattened to form the second dimension of the final
              matrix (the width of the matrix). As a result, height of the
              flattened matrix is equal to the product of $X$'s first
              `x_num_col_dims` dimensions' sizes, and width of the flattened
              matrix is equal to the product of $X$'s last `rank(x) - num_col_dims`
              dimensions' size. For example, suppose $X$ is a 6-dimensional
              tensor with the shape [2, 3, 4, 5, 6], and `x_num_col_dims` = 3.
              Thus, the flattened matrix will have a shape [2 x 3 x 4, 5 x 6] =
              [24, 30].
        )DOC")
        .SetDefault(1)
        .EqualGreaterThan(1);
    AddAttr<int>(
        "y_num_col_dims",
        R"DOC((int, default 1), The mul_op can take tensors with more than two,
              dimensions as its inputs. If the input $Y$ is a tensor with more
              than two dimensions, $Y$ will be flattened into a two-dimensional
              matrix first. The attribute `y_num_col_dims` determines how $Y$ is
              flattened. See comments of `x_num_col_dims` for more details.
        )DOC")
        .SetDefault(1)
        .EqualGreaterThan(1);
    AddAttr<bool>(
        "is_col_major",
        R"DOC((bool, default False), Is the memory arragement is column-major.
              Set true for col-major, false for row-major.
        )DOC")
        .SetDefault(false);
    AddAttr<bool>(
        "is_transpose_A",
        R"DOC((bool, default False), Does SPMMA transposes matrix A before computing.
        )DOC")
        .SetDefault(false);
    AddAttr<bool>(
        "is_transpose_B",
        R"DOC((bool, default False), Does SPMMA transposes matrix B before computing.
        )DOC")
        .SetDefault(false);
    AddAttr<int>(
        "m",
        R"DOC((int, optional), The m dimension of A (m, k) x B (n, k). default is X.dim[0].
        )DOC")
        .SetDefault(-1)
        .EqualGreaterThan(-1);
    AddAttr<int>(
        "n",
        R"DOC((int, optional), The n dimension of A (m, k) x B (n, k). default is Y.dim[1].
        )DOC")
        .SetDefault(-1)
        .EqualGreaterThan(-1);
    AddAttr<int>(
        "k",
      R"DOC((int, optional), The k dimension of A (m, k) x B (n, k). default is X.dim[1].
        )DOC")
        .SetDefault(-1)
        .EqualGreaterThan(-1);
    AddAttr<int>(
        "lda",
      R"DOC((int, optional), The leading dimension of A (m, k). default is n.
        )DOC")
        .SetDefault(-1)
        .EqualGreaterThan(-1);
    AddAttr<int>(
        "ldb",
      R"DOC((int, optional), The leading dimension of B (k, n). default is n.
        )DOC")
        .SetDefault(-1)
        .EqualGreaterThan(-1);
    AddAttr<int>(
        "ldc",
      R"DOC((int, optional), The leading dimension of C (m, n). default is n.
        )DOC")
        .SetDefault(-1)
        .EqualGreaterThan(-1);
    AddAttr<std::vector<int>>(
        "output_shape",
        "(std::vector<int>, optional) Target shape of output matrix."
        "default is (*x_dim[:x_num_col_dims], *x_dim[y_num_col_dims:])")
        .SetDefault({});
    AddComment(R"DOC(
MulSparse Operator.

This operator is used to perform sparse matrix multiplication for input $X$ and $Y$.

The equation is:

$$Out = Pruned(X) * Y$$

Both the input $X$ and $Y$ can carry the LoD (Level of Details) information,
or not. But the output only shares the LoD information with input $X$.

)DOC");
  }
};

class MulSparseGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "mul");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "mul");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "mul");
    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");

    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->SetOutputDim(y_grad_name, y_dims);
    }
  }
};

template <typename T>
class MulSparseOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("mul_grad");
    retv->SetInput("X", this->Input("X"));
    retv->SetInput("Y", this->Input("Y"));
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    retv->SetAttrMap(this->Attrs());
  }
};

class MulSparseOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Out"}};
    return m;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(mul_sparse, ops::MulSparseOp, ops::MulSparseOpMaker, ops::MulSparseOpInferVarType);

REGISTER_OP_CPU_KERNEL(
    mul_sparse, ops::MulKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MulKernel<paddle::platform::CPUDeviceContext, double>);
