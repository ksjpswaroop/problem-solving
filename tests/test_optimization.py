"""Phase 3: Optimization algorithm tests (GD, GA, SA)."""

import math


class TestGradientDescent:
    """Gradient Descent tests."""

    def test_minimize_quadratic(self) -> None:
        from problemsolving.optimization.gradient_descent import gd_solve

        # f(x) = (x-3)^2, minimum at x=3
        result = gd_solve(
            objective=lambda x: (x[0] - 3) ** 2,
            gradient=lambda x: [2 * (x[0] - 3)],
            initial=[0.0],
            learning_rate=0.1,
            max_iterations=100,
        )
        assert result is not None
        assert abs(result["solution"][0] - 3.0) < 0.01

    def test_minimize_2d_quadratic(self) -> None:
        from problemsolving.optimization.gradient_descent import gd_solve

        # f(x,y) = (x-1)^2 + (y-2)^2, minimum at (1, 2)
        result = gd_solve(
            objective=lambda x: (x[0] - 1) ** 2 + (x[1] - 2) ** 2,
            gradient=lambda x: [2 * (x[0] - 1), 2 * (x[1] - 2)],
            initial=[0.0, 0.0],
            learning_rate=0.1,
            max_iterations=200,
        )
        assert result is not None
        assert abs(result["solution"][0] - 1.0) < 0.01
        assert abs(result["solution"][1] - 2.0) < 0.01

    def test_convergence_tracked(self) -> None:
        from problemsolving.optimization.gradient_descent import gd_solve

        result = gd_solve(
            objective=lambda x: x[0] ** 2,
            gradient=lambda x: [2 * x[0]],
            initial=[10.0],
            learning_rate=0.1,
            max_iterations=50,
        )
        assert result is not None
        assert "iterations" in result
        assert result["iterations"] <= 50
        assert result["objective_value"] < 0.01

    def test_early_stop_on_tolerance(self) -> None:
        from problemsolving.optimization.gradient_descent import gd_solve

        result = gd_solve(
            objective=lambda x: x[0] ** 2,
            gradient=lambda x: [2 * x[0]],
            initial=[1.0],
            learning_rate=0.1,
            max_iterations=10000,
            tolerance=1e-6,
        )
        assert result is not None
        assert result["iterations"] < 10000


class TestGeneticAlgorithm:
    """Genetic Algorithm tests."""

    def test_maximize_simple(self) -> None:
        from problemsolving.optimization.genetic import ga_solve

        # Maximize f(x) = -(x-5)^2 + 25, peak at x=5 with value 25
        result = ga_solve(
            fitness=lambda ind: -(ind[0] - 5) ** 2 + 25,
            bounds=[(-10, 10)],
            population_size=50,
            generations=100,
        )
        assert result is not None
        assert abs(result["solution"][0] - 5.0) < 1.0
        assert result["fitness"] > 20

    def test_2d_optimization(self) -> None:
        from problemsolving.optimization.genetic import ga_solve

        # Maximize f(x,y) = -((x-3)^2 + (y-4)^2), peak at (3, 4)
        result = ga_solve(
            fitness=lambda ind: -((ind[0] - 3) ** 2 + (ind[1] - 4) ** 2),
            bounds=[(-10, 10), (-10, 10)],
            population_size=80,
            generations=150,
        )
        assert result is not None
        assert abs(result["solution"][0] - 3.0) < 1.5
        assert abs(result["solution"][1] - 4.0) < 1.5

    def test_returns_generation_count(self) -> None:
        from problemsolving.optimization.genetic import ga_solve

        result = ga_solve(
            fitness=lambda ind: -ind[0] ** 2,
            bounds=[(-10, 10)],
            population_size=20,
            generations=50,
        )
        assert result is not None
        assert "generations" in result
        assert result["generations"] <= 50

    def test_respects_bounds(self) -> None:
        from problemsolving.optimization.genetic import ga_solve

        result = ga_solve(
            fitness=lambda ind: ind[0],
            bounds=[(0, 5)],
            population_size=30,
            generations=50,
        )
        assert result is not None
        assert 0 <= result["solution"][0] <= 5


class TestSimulatedAnnealing:
    """Simulated Annealing tests."""

    def test_minimize_quadratic(self) -> None:
        import random

        from problemsolving.optimization.simulated_annealing import sa_solve

        random.seed(42)
        # Minimize f(x) = (x-7)^2
        result = sa_solve(
            objective=lambda x: (x[0] - 7) ** 2,
            initial=[0.0],
            neighbor_fn=lambda x: [x[0] + random.gauss(0, 0.5)],
            temperature=10.0,
            cooling_rate=0.995,
            max_iterations=5000,
        )
        assert result is not None
        assert abs(result["solution"][0] - 7.0) < 1.0

    def test_minimize_2d(self) -> None:
        import random

        from problemsolving.optimization.simulated_annealing import sa_solve

        random.seed(42)
        result = sa_solve(
            objective=lambda x: (x[0] - 2) ** 2 + (x[1] + 3) ** 2,
            initial=[0.0, 0.0],
            neighbor_fn=lambda x: [x[0] + random.gauss(0, 0.3), x[1] + random.gauss(0, 0.3)],
            temperature=10.0,
            cooling_rate=0.995,
            max_iterations=5000,
        )
        assert result is not None
        assert abs(result["solution"][0] - 2.0) < 1.5
        assert abs(result["solution"][1] + 3.0) < 1.5

    def test_escapes_local_minimum(self) -> None:
        import random

        from problemsolving.optimization.simulated_annealing import sa_solve

        random.seed(42)

        # Rastrigin-like: global min at x=0, many local minima
        def rastrigin_1d(x: list[float]) -> float:
            return x[0] ** 2 + 10 * (1 - math.cos(2 * math.pi * x[0]))

        result = sa_solve(
            objective=rastrigin_1d,
            initial=[3.0],
            neighbor_fn=lambda x: [x[0] + random.gauss(0, 0.5)],
            temperature=20.0,
            cooling_rate=0.999,
            max_iterations=10000,
        )
        assert result is not None
        # Should get close to global minimum at 0
        assert result["objective_value"] < 5.0

    def test_returns_metadata(self) -> None:
        import random

        from problemsolving.optimization.simulated_annealing import sa_solve

        random.seed(42)
        result = sa_solve(
            objective=lambda x: x[0] ** 2,
            initial=[5.0],
            neighbor_fn=lambda x: [x[0] + random.gauss(0, 0.5)],
            temperature=10.0,
            cooling_rate=0.99,
            max_iterations=1000,
        )
        assert result is not None
        assert "iterations" in result
        assert "objective_value" in result


class TestOptimizationComparison:
    """Cross-algorithm comparison tests."""

    def test_all_find_minimum_of_simple_quadratic(self) -> None:
        import random

        from problemsolving.optimization.genetic import ga_solve
        from problemsolving.optimization.gradient_descent import gd_solve
        from problemsolving.optimization.simulated_annealing import sa_solve

        random.seed(42)

        # All three should find minimum of f(x) = (x-5)^2 near x=5
        gd = gd_solve(
            objective=lambda x: (x[0] - 5) ** 2,
            gradient=lambda x: [2 * (x[0] - 5)],
            initial=[0.0],
            learning_rate=0.1,
            max_iterations=100,
        )

        ga = ga_solve(
            fitness=lambda ind: -((ind[0] - 5) ** 2),
            bounds=[(-10, 20)],
            population_size=50,
            generations=100,
        )

        sa = sa_solve(
            objective=lambda x: (x[0] - 5) ** 2,
            initial=[0.0],
            neighbor_fn=lambda x: [x[0] + random.gauss(0, 0.5)],
            temperature=10.0,
            cooling_rate=0.995,
            max_iterations=5000,
        )

        assert gd is not None
        assert ga is not None
        assert sa is not None
        # GD should be most precise (exact gradient)
        assert abs(gd["solution"][0] - 5.0) < 0.1
        # GA and SA should be within reasonable range
        assert abs(ga["solution"][0] - 5.0) < 2.0
        assert abs(sa["solution"][0] - 5.0) < 2.0
