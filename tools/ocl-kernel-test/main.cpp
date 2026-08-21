
#include <cstdio>
#include <cstring>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-ocl.h"

#include "test-op.h"

static void usage() {
    printf("usage: ocl-op-test <op-name|all> [--seed=N]\n");
    printf("\navailable ops:\n");
    for (const auto & t : ocl_op_test_registry()) {
        printf("  %s\n", t.name);
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }

    const std::string filter = argv[1];
    if (filter == "--help" || filter == "-h") {
        usage();
        return 0;
    }

    uint32_t seed = 42;
    for (int i = 2; i < argc; i++) {
        if (strncmp(argv[i], "--seed=", 7) == 0) {
            seed = (uint32_t) atoi(argv[i] + 7);
        }
    }

    // 被测 backend (ggml-ocl) 与参考 backend (ggml-cpu)
    ggml_backend_reg_t reg = ggml_backend_ocl_reg();
    GGML_ASSERT(reg != nullptr && ggml_backend_reg_dev_count(reg) > 0);
    ggml_backend_t backend_ocl = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
    GGML_ASSERT(backend_ocl != nullptr);

    ggml_backend_t backend_cpu = ggml_backend_cpu_init();
    GGML_ASSERT(backend_cpu != nullptr);

    int n_run = 0, n_pass = 0;
    for (const auto & t : ocl_op_test_registry()) {
        if (filter != "all" && filter != t.name) {
            continue;
        }
        printf("\n=== op: %s ===\n", t.name);
        n_run++;
        n_pass += t.run(backend_ocl, backend_cpu, seed) ? 1 : 0;
    }

    ggml_backend_free(backend_cpu);
    ggml_backend_free(backend_ocl);

    printf("\n%d/%d ops executed\n", n_pass, n_run);
    return n_run == n_pass ? 0 : 1;
}
