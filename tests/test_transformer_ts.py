"""Tests for Transformer Time Series module."""
import pytest
import numpy as np


class TestTFT:
    def test_tft_init(self):
        from oracle.prediction.transformer_ts import TFTPredictor
        tft = TFTPredictor(d_model=16, n_heads=2, seed=42)
        assert tft.d_model == 16
        assert tft.n_heads == 2
        assert not tft._fitted

    def test_tft_fit_and_forecast(self):
        from oracle.prediction.transformer_ts import TFTPredictor
        np.random.seed(42)
        data = 100 + np.cumsum(np.random.normal(0, 1, 100))
        tft = TFTPredictor(d_model=16, n_heads=2, n_layers=1, seed=42)
        tft.fit(data, n_epochs=10)
        result = tft.forecast(n_ahead=5)
        assert len(result.forecast) == 5
        assert len(result.forecast_lower) == 5
        assert len(result.forecast_upper) == 5
        assert result.method == "transformer_tft"
        assert result.n_parameters > 0

    def test_tft_attention_weights(self):
        from oracle.prediction.transformer_ts import TFTPredictor
        np.random.seed(42)
        data = 100 + np.cumsum(np.random.normal(0, 1, 100))
        tft = TFTPredictor(d_model=16, n_heads=2, seed=42)
        tft.fit(data, n_epochs=5)
        result = tft.forecast(n_ahead=3)
        assert len(result.attention_weights) == 3

    def test_tft_loss_history(self):
        from oracle.prediction.transformer_ts import TFTPredictor
        np.random.seed(42)
        data = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.3, 100)
        tft = TFTPredictor(d_model=16, n_heads=2, seed=42)
        tft.fit(data, n_epochs=10)
        assert len(tft.loss_history) == 10
        assert all(np.isfinite(l) for l in tft.loss_history)

    def test_tft_interpret(self):
        from oracle.prediction.transformer_ts import TFTPredictor
        np.random.seed(42)
        data = 100 + np.cumsum(np.random.normal(0, 1, 100))
        tft = TFTPredictor(d_model=16, n_heads=2, seed=42)
        tft.fit(data, n_epochs=5)
        interp = tft.interpret()
        assert "variable_importance" in interp
        assert "input_attention" in interp
        assert "n_layers" in interp
        assert interp["n_layers"] == 2

    def test_tft_attention_pattern(self):
        from oracle.prediction.transformer_ts import TFTPredictor
        np.random.seed(42)
        data = 100 + np.cumsum(np.random.normal(0, 1, 100))
        tft = TFTPredictor(d_model=16, n_heads=2, seed=42)
        tft.fit(data, n_epochs=5)
        pattern = tft.attention_pattern(data)
        assert "attention_matrix" in pattern
        assert "sequence_length" in pattern

    def test_tft_not_fitted_raises(self):
        from oracle.prediction.transformer_ts import TFTPredictor
        tft = TFTPredictor()
        with pytest.raises(ValueError):
            tft.forecast(n_ahead=5)

    def test_tft_not_fitted_interpret_raises(self):
        from oracle.prediction.transformer_ts import TFTPredictor
        tft = TFTPredictor()
        with pytest.raises(ValueError):
            tft.interpret()


class TestPositionalEncoding:
    def test_encoding(self):
        from oracle.prediction.transformer_ts import PositionalEncoding
        pe = PositionalEncoding(d_model=32)
        x = np.zeros((10, 32))
        result = pe.encode(x)
        assert result.shape == (10, 32)
        assert not np.allclose(result, 0)

    def test_encoding_values(self):
        from oracle.prediction.transformer_ts import PositionalEncoding
        pe = PositionalEncoding(d_model=16, max_len=100)
        x = np.zeros((5, 16))
        result = pe.encode(x)
        assert result.shape == (5, 16)
        pos_0 = result[0]
        pos_1 = result[1]
        assert not np.allclose(pos_0, pos_1)


class TestMultiHeadAttention:
    def test_attention(self):
        from oracle.prediction.transformer_ts import MultiHeadAttention
        mha = MultiHeadAttention(d_model=16, n_heads=2)
        x = np.random.randn(10, 16)
        output, weights = mha.forward(x, x, x)
        assert output.shape == (10, 16)
        assert weights.shape[0] == 10
        assert weights.shape[1] == 10


class TestGRN:
    def test_grn(self):
        from oracle.prediction.transformer_ts import GatedResidualNetwork
        grn = GatedResidualNetwork(d_model=16)
        x = np.random.randn(10, 16)
        output = grn.forward(x)
        assert output.shape == (10, 16)


class TestTransformerResult:
    def test_result_dataclass(self):
        from oracle.prediction.transformer_ts import TransformerResult
        result = TransformerResult(
            forecast=[1.0, 2.0, 3.0],
            forecast_lower=[0.5, 1.5, 2.5],
            forecast_upper=[1.5, 2.5, 3.5],
            attention_weights=[[0.1, 0.2]],
            variable_importance=[0.3, 0.7],
            loss_history=[0.5],
            n_epochs=1,
            n_parameters=100,
        )
        assert len(result.forecast) == 3
        assert result.method == "transformer_tft"