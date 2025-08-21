import pytest
from src.config import Config

def test_config_loading():
    config = Config()
    params = config.get_params()
    
    assert isinstance(params, dict)
    assert "num_years" in params
    assert "initial_savings" in params
    assert "total_land" in params
    assert "temperature" in params
    assert "rainfall" in params
    assert "crops" in params
    assert isinstance(params["crops"], list)

def test_random_instance_generation():
    config = Config()
    instances = config.generate_random_instances(10)
    
    assert len(instances) == 10
    for instance in instances:
        assert instance["temperature"] != config.get_params()["temperature"]  # Should be perturbed
        assert instance["rainfall"] != config.get_params()["rainfall"]  # Should be perturbed
        for crop in instance["crops"]:
            assert "price" in crop
            assert "yield" in crop