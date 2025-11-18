from hydra_zen import store

assets_test1 = ["3QG", "3QJ"]
assets_test2 = ["3QM", "3QP"]
asset_store = store(group="assets")
asset_store(assets_test1, name="asset1")
asset_store(assets_test2, name="asset2")
