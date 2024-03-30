#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

VectorXd solveWithPALU(const MatrixXd& A, const VectorXd& b) {
    PartialPivLU<MatrixXd> palu(A);
    VectorXd x = palu.solve(b);

    cout << "Solution using PALU decomposition:\n" << x << endl;
    return x;
}

VectorXd solveWithQR(const MatrixXd& A, const VectorXd& b) {
    ColPivHouseholderQR<MatrixXd> qr(A);
    VectorXd x = qr.solve(b);
    cout << "Solution using QR decomposition:\n" << x << endl;
    return x;
}

void checkRelativeError(const VectorXd& x, const VectorXd& y, const VectorXd& z) {

    double rel_err_palu = (x - y).norm() / x.norm();
    double rel_err_qr = (x-z).norm()/x.norm();
    cout << "Relative error PALU: " << rel_err_palu << endl;
    cout << "Relative error QR: " << rel_err_qr << endl;
}

int main() {
    // System 1
    MatrixXd A1(2, 2);
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
        8.320502943378437e-01, -9.992887623566787e-01;
    VectorXd b1(2);
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;
    VectorXd x1(2);
    x1 << -1.0e+0, -1.0e+00;

    cout << "System 1:\n";
    VectorXd x1_palu = solveWithPALU(A1, b1);
    VectorXd x1_qr = solveWithQR(A1, b1);
    checkRelativeError( x1,x1_palu,x1_qr);
    cout << endl;

    // System 2
    MatrixXd A2(2, 2);
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
        8.320502943378437e-01, -8.324762492991313e-01;
    VectorXd b2(2);
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;

    cout << "System 2:\n";
    VectorXd x2_palu = solveWithPALU(A2, b2);
    VectorXd x2_qr = solveWithQR(A2, b2);
    checkRelativeError( x1,x2_palu, x2_qr);
    cout << endl;

    // System 3
    MatrixXd A3(2, 2);
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
        8.320502943378437e-01, -8.320502947645361e-01;
    VectorXd b3(2);
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;

    cout << "System 3:\n";
    VectorXd x3_palu= solveWithPALU(A3, b3);
    VectorXd x3_qr = solveWithQR(A3, b3);
    checkRelativeError( x1, x3_palu, x3_qr);
    cout << endl;

    return 0;
}
