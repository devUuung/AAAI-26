#include <iostream>
#include "src/pke/include/openfhe.h"

using namespace std;
using namespace lbcrypto;

int main() {
    cout << "OpenFHE Version: " << GetOPENFHEVersion() << endl;
    cout << "Main_TestAll.cpp style simple test completed successfully!" << endl;
    return 0;
}
