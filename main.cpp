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
  
  auto mesh = std::make_shared<RectangleMesh>(MPI_COMM_WORLD,
                                              Point (0, -M_PI/2), Point (M_PI, M_PI/2),
                                              1, 1, "crossed");
  auto W = std::make_shared<LinearKirchhoff::Form_dkt_FunctionSpace_0>(mesh);
  auto Theta = std::make_shared<LinearKirchhoff::Form_p22_FunctionSpace_0>(mesh);
  
  auto u0 = std::make_shared<Constant>(0.0);
  auto boundary = std::make_shared<DirichletBoundary>();

  // HermiteDirichletBC bc(W, u0, boundary);

  LinearKirchhoff::Form_dkt a(W, W);
  LinearKirchhoff::Form_force L(W);
  
  LinearKirchhoff::Form_p22 p22(Theta, Theta);

  auto f = std::make_shared<Force>();
  L.f = f;

  Function u(W);

  PETScMatrix A;
  KirchhoffAssembler assembler;
  assembler.assemble(A, a, p22);
    
  return 1;
}
