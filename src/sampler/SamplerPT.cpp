#include "Utils.h"
#include "SamplerPT.h"
#include "SamplerPTChain.h"
#include "VariableSet.h"

namespace bcm3 {

	SamplerPT::SamplerPT(size_t threads, size_t max_memory_use)
		: Sampler(threads, max_memory_use)
		, blocking_strategy("one_block")
		, proposal_type("gaussian_mixture")
		, proposal_transform_to_unbounded(false)
		, output_proposal_adaptation(true)
		, swapping_scheme(ESwappingScheme::StochasticEvenOdd)
		, exchange_probability(0.5)
		, num_exploration_steps(1)
		, adapt_proposal_samples(2000)
		, adapt_proposal_times(2)
		, max_history_size(2000)
		, adapt_proposal_max_history_samples(2000)
		, adapt_proposal_max_clustering_samples(1000)
		, stop_proposal_scaling(6000)
		, history_clustering_nn(3)
		, history_clustering_nn2(7)
		, history_clustering_nclusters(4)
		, proposal_scaling_learning_rate(0.05)
		, proposal_scaling_ema_period(2000)
		, proposal_adaptations_done(0)
		, proposal_scaling_adaptations_done(false)
		, previous_swap_even(false)
		, target_acceptance_rate(0.234)
		, proposal_t_dof(0.0)
	{
	}

	SamplerPT::~SamplerPT()
	{
	}

	bool SamplerPT::LoadSettings(const boost::program_options::variables_map& vm)
	{
		if (!Sampler::LoadSettings(vm)) {
			return false;
		}

		try {
			blocking_strategy = vm["ptmhsampler.blocking_strategy"].as<std::string>();
			proposal_type = vm["ptmhsampler.proposal_type"].as<std::string>();
			proposal_transform_to_unbounded = vm["ptmhsampler.proposal_transform_to_unbounded"].as<bool>();
			max_history_size = vm["ptmhsampler.max_history_size"].as<size_t>();
			adapt_proposal_samples = vm["ptmhsampler.adapt_proposal_samples"].as<size_t>();
			stop_proposal_scaling = vm["ptmhsampler.stop_proposal_scaling"].as<size_t>();
			history_clustering_nn = vm["ptmhsampler.sample_clustering_kernel_nn"].as<size_t>();
			history_clustering_nn2 = vm["ptmhsampler.sample_clustering_kernel_nn2"].as<size_t>();
			history_clustering_nclusters = vm["ptmhsampler.sample_clustering_num_clusters"].as<size_t>();
			adapt_proposal_times = vm["ptmhsampler.adapt_proposal_times"].as<size_t>();
			adapt_proposal_max_history_samples = vm["ptmhsampler.adapt_proposal_max_history_samples"].as<size_t>();
			adapt_proposal_max_clustering_samples = vm["ptmhsampler.adapt_proposal_max_clustering_samples"].as<size_t>();
			exchange_probability = vm["ptmhsampler.exchange_probability"].as<Real>();
			num_exploration_steps = vm["ptmhsampler.num_exploration_steps"].as<size_t>();
			output_proposal_adaptation = vm["ptmhsampler.output_proposal_adaptation"].as<bool>();
			proposal_t_dof = vm["ptmhsampler.proposal_t_dof"].as<Real>();

			std::string swapping_scheme_str = vm["ptmhsampler.swapping_scheme"].as<std::string>();
			if (swapping_scheme_str == "stochastic_random") {
				swapping_scheme = ESwappingScheme::StochasticRandom;
			} else if (swapping_scheme_str == "stochastic_even_odd") {
				swapping_scheme = ESwappingScheme::StochasticEvenOdd;
			} else if (swapping_scheme_str == "deterministic_even_odd") {
				swapping_scheme = ESwappingScheme::DeterministicEvenOdd;
			} else {
				LOGERROR("Unknown swapping scheme \"%s\"", swapping_scheme_str.c_str());
				return false;
			}

			if (adapt_proposal_samples > 0) {
				proposal_scaling_ema_period = (size_t)ceil(adapt_proposal_samples * use_every_nth * (1.0 - exchange_probability) / 10);
				proposal_scaling_learning_rate = pow(100.0, 1.0 / proposal_scaling_ema_period) - 1.0;
			}

			size_t num_chains = vm["ptmhsampler.num_chains"].as<size_t>();
			Real temperature_power = vm["ptmhsampler.temperature_schedule_power"].as<Real>();
			Real temperature_max = vm["ptmhsampler.temperature_schedule_max"].as<Real>();

			// Power-law scaling from 1.0 downwards and the lowest chain at temperature 0
			temperatures.setZero(num_chains);
			for (size_t i = 1; i < num_chains - 1; i++) {
				Real alpha = i / (Real)(num_chains - 1);
				temperatures(i) = temperature_max * pow(alpha, temperature_power);
			}
			temperatures(num_chains - 1) = temperature_max;
		} catch (boost::program_options::error& e) {
			LOGERROR("Error parsing sampler: %s", e.what());
			return false;
		}

		return true;
	}

	bool SamplerPT::Initialize()
	{
		if (!Sampler::Initialize()) {
			return false;
		}

		if (temperatures.size() == 0) {
			LOGERROR("Temperatures have not been specified.");
			return false;
		}

		// Calculate the number of samples we can store in history
		size_t expected_sample_history = adapt_proposal_samples * use_every_nth;
		if (temperatures.size() > 1 && swapping_scheme == ESwappingScheme::DeterministicEvenOdd) {
			expected_sample_history *= (num_exploration_steps + 1);
		}
		size_t history_subsampling = 1;
		size_t sample_history = expected_sample_history;
		if (sample_history > max_history_size) {
			history_subsampling = (expected_sample_history + max_history_size - 1) / max_history_size;
			sample_history = expected_sample_history / history_subsampling;
		}
		if (adapt_proposal_times > 0) {
			BCMLOG("Using history size %zd with 1 in %zd subsampling for proposal adaptation", sample_history, history_subsampling);
		}

		// Initialize chains
		chains.resize(temperatures.size());
		for (size_t i = 0; i < chains.size(); i++) {
			chains[i] = std::make_unique<SamplerPTChain>(this);
			chains[i]->SetTemperature(temperatures(i), true);
			bool result = chains[i]->Initialize(sample_history, history_subsampling);
			if (!result) {
				return false;
			}
		}
		previous_swap_even = false;

		// Initialize proposal adaptations
		proposal_adaptations_done = 0;
		proposal_scaling_adaptations_done = false;

		return true;
	}

	void SamplerPT::AddOptionsDescription(boost::program_options::options_description& pod)
	{
		pod.add_options()
			("ptmhsampler.num_chains",								boost::program_options::value<size_t>()->default_value(6),								"Number of parallel-tempered chains")
			("ptmhsampler.blocking_strategy",						boost::program_options::value<std::string>()->default_value("one_block"),				"Blocking strateg: one_block, no_blocking, Turek, clustered_autoblock")
			("ptmhsampler.proposal_type",							boost::program_options::value<std::string>()->default_value("gaussian_mixture"),		"Type of proposal: global_covariance, clustered_covariance, gaussian_mixture, gaussian_mixture_fit_in_r")
			("ptmhsampler.proposal_transform_to_unbounded",			boost::program_options::value<bool>()->default_value(false),							"Specifies whether to transform variables that have a bounded prior to an unbounded domain before applying the proposal")
			("ptmhsampler.adapt_proposal_samples",					boost::program_options::value<size_t>()->default_value(2000),							"Number of samples after which the proposal distribution should be adapted, 0 for no adaptation (after thinning).")
			("ptmhsampler.adapt_proposal_times",					boost::program_options::value<size_t>()->default_value(2),								"Number of times the proposal variance should be adapted.")
			("ptmhsampler.max_history_size",						boost::program_options::value<size_t>()->default_value(2000),							"Maximum number of samples to store in the sample history (before thinning).")
			("ptmhsampler.adapt_proposal_max_history_samples",		boost::program_options::value<size_t>()->default_value(2000),							"Maximum number of samples to use in the GMM/covariance fitting (before thinning).")
			("ptmhsampler.adapt_proposal_max_clustering_samples",	boost::program_options::value<size_t>()->default_value(1000),							"Maximum number of samples to use in the spectral clustering (before thinning).")
			("ptmhsampler.stop_proposal_scaling",					boost::program_options::value<size_t>()->default_value(6000),							"Stop adaptive scaling of the proposal distribution after this many samples (after thinning).")
			("ptmhsampler.sample_clustering_kernel_nn",				boost::program_options::value<size_t>()->default_value(3),								"Sample history clustering: nn-value of the density-aware kernel.")
			("ptmhsampler.sample_clustering_kernel_nn2",			boost::program_options::value<size_t>()->default_value(7),								"Sample history clustering: nn2-value of the density-aware kernel.")
			("ptmhsampler.sample_clustering_num_clusters",			boost::program_options::value<size_t>()->default_value(4),								"Sample history clustering: number of clusters.")
			("ptmhsampler.swapping_scheme",							boost::program_options::value<std::string>()->default_value("deterministic_even_odd"),	"Swapping scheme, can be either stochastic_random, stochastic_even_odd or deterministic_even_odd.")
			("ptmhsampler.exchange_probability",					boost::program_options::value<Real>()->default_value(0.5),								"Probability of choosing an exchange move instead of a mutate move (only used if swapping_scheme is stochastic).")
			("ptmhsampler.num_exploration_steps",					boost::program_options::value<size_t>()->default_value(1),								"Number of exploration steps between swaps (only used if swapping_scheme is deterministic).")
			("ptmhsampler.temperature_schedule_power",				boost::program_options::value<Real>()->default_value(3.0),								"Specifies the rate of the power-law used for the fixed/initial temperature schedule.")
			("ptmhsampler.temperature_schedule_max",				boost::program_options::value<Real>()->default_value(1.0),								"Specifies the maximum temperature used for the fixed/initial temperature schedule.")
			("ptmhsampler.output_proposal_adaptation",				boost::program_options::value<bool>()->default_value(false),							"Whether to output information describing the proposal adaptation.")
			("ptmhsampler.proposal_t_dof",							boost::program_options::value<Real>()->default_value(0.0),								"Degrees of freedom of the t-distribution proposal, set to 0 to use a standard normal distribution.")
			;
	}

	bool SamplerPT::RunImpl()
	{
		bool result = true;

		BCMLOG("Finding starting point...");
		for (auto& chain : chains) {
			result = chain->FindStartingPosition();
			if (!result) {
				return false;
			}
		}
	
		BCMLOG("Starting sampling loop...");
		size_t total_samples = num_samples * use_every_nth;
		Real max_lposterior = -std::numeric_limits<Real>::infinity();
		for (size_t si = 0; si < total_samples; si++) {
			UpdateProgress(si / (Real)total_samples, false);

			size_t sample_ix = si / use_every_nth;

			if (chains.size() > 1) {
				if (swapping_scheme == ESwappingScheme::StochasticRandom || swapping_scheme == ESwappingScheme::StochasticEvenOdd) {
					// Randomly choose between mutate and exchange moves
					Real exchange = rng.GetReal();
					if (exchange < exchange_probability) {
						DoExchangeMove(si);
					} else {
						result = DoMutateMove();
					}
				} else if (swapping_scheme == ESwappingScheme::DeterministicEvenOdd) {
					DoExchangeMove(si);

					for (size_t ei = 0; ei < num_exploration_steps; ei++) {
						result = DoMutateMove();
						if (!result) {
							break;
						}
					}
				}
			} else {
				result = DoMutateMove();
			}

			if (!result) {
				LOGERROR("Sample step failed");
				LogStatistics();
				return false;
			}

			if ((*chains.rbegin())->GetLogPowerPosterior() > max_lposterior) {
				max_lposterior = (*chains.rbegin())->GetLogPowerPosterior();
				BCMLOG("New max log posterior: %g", max_lposterior);
			}

			if ((si + 1) % use_every_nth == 0) {
				EmitSample(sample_ix);

				if (adapt_proposal_samples > 0 && ((sample_ix + 1) % adapt_proposal_samples == 0) && si + 1 != total_samples && proposal_adaptations_done < adapt_proposal_times) {
					// Force the progress update
					UpdateProgress(si / (Real)total_samples, true);

					LogStatistics();
					BCMLOG("Updating proposal...");
					bool result = true;
					for (auto& chain : chains) {
						chain->AdaptProposalAsync();
					}
					for (auto& chain : chains) {
						result &= chain->AdaptProposalWait();
					}
					if (!result) {
						LOGERROR("Proposal adaptation failed");
						return false;
					}
					proposal_adaptations_done++;
				}
				if (stop_proposal_scaling > 0 && sample_ix > stop_proposal_scaling) {
					if (!proposal_scaling_adaptations_done) {
						BCMLOG("Stopping adaptation of proposal scale");
						proposal_scaling_adaptations_done = true;
					}
				}
			}
		}

		return true;
	}

	void SamplerPT::LogStatistics()
	{
		// Statistics
		BCMLOG("");
		BCMLOG("Acceptance statistics:");
		BCMLOG("Temperature | Mutate (all) | Exchange (all)");
		for (const auto& chain : chains) {
			chain->LogStatistics();
		}

		BCMLOG("");
		BCMLOG("Proposal info:");
		(*chains.rbegin())->LogProposalInfo();
	}

	void SamplerPT::DoExchangeMove(size_t sample_ix)
	{
		if (swapping_scheme == ESwappingScheme::StochasticEvenOdd || swapping_scheme == ESwappingScheme::DeterministicEvenOdd) {
			// Alternatingly swap the odd and even chains with the subsequent chains
			size_t start_ix;
			if (previous_swap_even) {
				start_ix = 1;
				previous_swap_even = false;
			} else {
				start_ix = 0;
				previous_swap_even = true;
			}

			size_t ci = start_ix;
			for (; ci < chains.size(); ci += 2) {
				SamplerPTChain& chain1 = *chains[ci];
				size_t ix2 = ci + 1;
				if (ix2 == chains.size()) {
					ix2 = 0;
				}
				SamplerPTChain& chain2 = *chains[ix2];
				bool exchanged = chain1.ExchangeMove(chain2);
			}
		} else if (swapping_scheme == ESwappingScheme::StochasticRandom) {
			size_t ci = rng.GetUnsignedInt(chains.size()-2);
			SamplerPTChain& chain1 = *chains[ci];
			SamplerPTChain& chain2 = *chains[ci+1];
			chain1.ExchangeMove(chain2);
		}
	}

	bool SamplerPT::DoMutateMove()
	{
		bool result = true;
		for (auto& chain : chains) {
			chain->MutateMoveAsync();
		}
		for (auto& chain : chains) {
			result &= chain->MutateMoveWait();
		}

		return result;
	}

	void SamplerPT::EmitSample(size_t sample_ix)
	{
		for (auto handler : sample_handlers) {
			for (auto& chain : chains) {
				if (chain->GetIsFixedTemperature()) {
					handler->ReceiveSample(chain->GetVarValues(), chain->GetLogPrior(), chain->GetLogLikelihood(), chain->GetTemperature(), 1.0);
				}
			}
		}
	}

}
