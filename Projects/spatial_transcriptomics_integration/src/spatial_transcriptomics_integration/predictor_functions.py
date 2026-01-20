def make_shared_gene_predictor(
    PredictorCls: Type[predictors.PredictorProtocol],
    predictor_kwargs: Dict[str, Any] | None = None,
) -> Callable[[AnnData, AnnData], AnnData]:
    """
    Factory for predictors that:
      - train on reference shared genes
      - predict reference-only genes in query

    Assumes PredictorCls has:
        .fit(X_shared_ref, X_ref_only)
        .predict(X_shared_query)
    """
    predictor_kwargs = predictor_kwargs or {}

    def predictor(query: AnnData, reference: AnnData) -> AnnData:
        shared_genes = np.intersect1d(query.var_names, reference.var_names)
        ref_only_genes = np.setdiff1d(reference.var_names, shared_genes)

        model = PredictorCls(**predictor_kwargs)

        X_ref_shared = reference[:, shared_genes].X
        X_ref_only = reference[:, ref_only_genes].X
        X_query_shared = query[:, shared_genes].X

        X_recon = model.fit(X_ref_shared, X_ref_only).predict(X_query_shared)

        return ad.AnnData(
            X=X_recon,
            obs=query.obs.copy(),
            var=reference[:, ref_only_genes].var.copy(),
        )

    return predictor

PREDICTOR_MAPPING = [
   exp_runner.Variable(
       make_shared_gene_predictor(n2l.pd.NmfPredictor, {"n_components": 3}),
       {
           "predictor_id": 0,
           "predictor_name": "nmf_3",
           "architecture": "NMF",
           "n_components": 3,
       },
   ),
   exp_runner.Variable(
       make_shared_gene_predictor(n2l.pd.NmfPredictor, {"n_components": 8}),
       {
           "predictor_id": 1,
           "predictor_name": "nmf_8",
           "architecture": "NMF",
           "n_components": 8,
       },
   ),
   exp_runner.Variable(
       make_shared_gene_predictor(n2l.pd.NmfPredictor, {"n_components": "auto"}),
       {
           "predictor_id": 2,
           "predictor_name": "nmf_auto",
           "architecture": "NMF",
           "n_components": 8,
       },
   ),
   exp_runner.Variable(
       make_shared_gene_predictor(n2l.pd.TangramPredictor),
       {"predictor_id": 3, "predictor_name": "tangram", "architecture": "Tangram"},
   ),
   exp_runner.Variable(
       make_shared_gene_predictor(
           partial(
               n2l.pd.VaePredictor,
               vae_cls=n2l.pd.models.VAE,
               vae_kwargs={
                   "hidden_features_in": 128,
                   "hidden_features_out": 128,
                   "latent_features": 3,
                   "lr": 1e-3,
               },
               devices=1,
           )
       ),
       {
           "predictor_id": 4,
           "predictor_name": "VAE_3",
           "architecture": "VAE",
           "hidden_features_in": 128,
           "hidden_features_out": 128,
           "latent_features": 3,
       },
   ),
   exp_runner.Variable(
       make_shared_gene_predictor(
           partial(
               n2l.pd.VaePredictor,
               vae_cls=n2l.pd.models.LDVAE,
               vae_kwargs={
                   "hidden_features_in": 128,
                   "latent_features": 3,
                   "lr": 1e-3,
               },
               devices=1,
           )
       ),
       {
           "predictor_id": 4,
           "predictor_name": "LDVAE_3",
           "architecture": "LDVAE",
           "hidden_features_in": 128,
           "latent_features": 3,
       },
   ),
   exp_runner.Variable(
       make_shared_gene_predictor(
           partial(
               n2l.pd.VaePredictor,
               vae_cls=n2l.pd.models.LEVAE,
               vae_kwargs={
                   "hidden_features_out": 128,
                   "latent_features": 3,
                   "lr": 1e-3,
               },
               devices=1,
           )
       ),
       {
           "predictor_id": 4,
           "predictor_name": "LEVAE_3",
           "architecture": "LEVAE",
           "hidden_features_out": 128,
           "latent_features": 3,
       },
   ),
   exp_runner.Variable(
       make_shared_gene_predictor(
           partial(
               n2l.pd.VaePredictor,
               vae_cls=n2l.pd.models.LVAE,
               vae_kwargs={
                   "latent_features": 3,
                   "lr": 1e-3,
               },
               devices=1,
           )
       ),
       {
           "predictor_id": 4,
           "predictor_name": "LVAE_3",
           "architecture": "LVAE",
           "latent_features": 3,
       },
   ),
]
