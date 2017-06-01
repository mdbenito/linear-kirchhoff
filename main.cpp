#include <iostream>
#include <iomanip>
#include <numeric>
#include <vector>
#include <dolfin.h>

#include "KirchhoffAssembler.h"
#include "LinearKirchhoff.h"

using namespace dolfin;

class Force : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = -9.8;  // TODO put some sensible value here
  }
  
};

class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary;
  }
};

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

    std::cout << "rows: " << num_rows << ", cols: " << num_cols << "\n";
    
    // la_index rows[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14};
    // la_index cols[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14};
    // double* block = new double[15*15];

    A.get(block.data(), num_rows, rows.data(), num_cols, cols.data());

    std::cout << std::setprecision(14);
    for (int i = 0; i < num_rows; i++) {
      for (int j = 0; j < num_cols; j++) {
        std::cout << " " << block[i*num_cols + j];
      }
      std::cout << "\n";
    }
}


/* 
 *
 */
int
main(void)
{
  // WTF?? Someone tries to access this global parameter later.
  // I have to add it to the global parameter list or dolfin
  // throws a runtime error.
  dolfin::parameters.add("num_threads", 0);
  
  // testDKT();
  //RectangleMesh(Point(0,-pi/2), Point(pi, pi/2), nx, ny, 'crossed')
  Point p0(0, -M_PI/2);
  Point p1(M_PI, M_PI/2);
  auto mesh = std::make_shared<RectangleMesh>(MPI_COMM_WORLD,
                                              p0, p1, 1, 1, "crossed");
  auto W = std::make_shared<LinearKirchhoff::Form_dkt_FunctionSpace_0>(mesh);
  auto Theta = std::make_shared<LinearKirchhoff::Form_p22_FunctionSpace_0>(mesh);
  
  auto u0 = std::make_shared<Constant>(0.0);
  auto boundary = std::make_shared<DirichletBoundary>();
  // HermiteDirichletBC bc(W, u0, boundary);

  LinearKirchhoff::Form_dkt a(W, W);
  LinearKirchhoff::Form_force L(W);
  auto b = std::make_shared<LinearKirchhoff::Form_p22>(Theta, Theta);

  auto f = std::make_shared<Force>();
  L.f = f;

  Function u(W);

  if (true) {
    KirchhoffAssembler assembler(b);
    EigenMatrix A;
    assembler.assemble(A, a);
    dump_full_tensor(A);

  } else {

    PETScMatrix B;
    Assembler dolfinAssembler;
    dolfinAssembler.assemble(B, *b);
    dump_full_tensor(B);
  }
  
  return 1;
}  
