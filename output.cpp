#include <iostream>
#include <iomanip>

#include <dolfin.h>
#include <Eigen/Dense>

using namespace dolfin;


// I'm gonna have to specialize this at some point...
template<typename mat_t>
void
dump_eigen(const mat_t& A)
{
  std::cout << std::setprecision(4);
  for (std::size_t i = 0; i < A.rows(); ++i) {
    for (std::size_t j = 0; j < A.cols(); ++j) {
      std::cout << std::setw(8) << A(i,j);
    }
    std::cout << std::endl;
  }
}

// template<typename T>
void
dump_full_tensor(const GenericMatrix& A)
{
    auto num_rows = A.size(0);
    auto num_cols = A.size(1);

    std::vector<la_index> rows(num_rows);
    std::vector<la_index> cols(num_cols);
    std::iota(rows.begin(), rows.end(), 0);
    std::iota(cols.begin(), cols.end(), 0);
    
    std::vector<double> block(num_rows*num_cols);

    // std::cout << "rows: " << num_rows << ", cols: " << num_cols << "\n";
    
    A.get(block.data(), num_rows, rows.data(), num_cols, cols.data());

    std::cout << std::setprecision(14);
    for (int i = 0; i < num_rows; i++) {
      for (int j = 0; j < num_cols-1; j++) {
        std::cout << block[i*num_cols + j] << " ";
      }
      // Don't print a trailing space, it confuses numpy.loadtxt()
      std::cout << block[(i+1)*num_cols - 1] << std::endl;
    }
}

