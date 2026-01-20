from spatial_transcriptomics_integration.experiments import gene_pred_benchmark


def test_experiments_init_exports_gene_pred_benchmark() -> None:
    assert callable(gene_pred_benchmark)
