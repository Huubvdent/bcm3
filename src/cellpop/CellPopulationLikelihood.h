#pragma once

#include "Experiment.h"
#include "Likelihood.h"
#include "RNG.h"
#include "SBMLModel.h"

class SBMLModel;

class CellPopulationLikelihood : public bcm3::Likelihood
{
public:
	CellPopulationLikelihood(size_t sampling_threads, size_t evaluation_threads);
	~CellPopulationLikelihood();
	
	virtual bool Initialize(std::shared_ptr<const bcm3::VariableSet> varset, boost::property_tree::ptree likelihood_node);
	virtual bool AddNonSampledParameters(const std::vector<std::string>& variable_names);
	virtual void SetNonSampledParameters(const VectorReal& values);
	virtual bool PostInitialize();
	virtual bool IsReentrant() { return false; }
	virtual void OutputEvaluationStatistics(const std::string& path) const;
	virtual bool EvaluateLogProbability(size_t threadix, const VectorReal& values, Real& logp);

	const Experiment* GetExperiment(const std::string& experiment);

private:
	// Static variables
	size_t sampling_threads;
	size_t evaluation_threads;
	std::shared_ptr<const bcm3::VariableSet> varset;

	std::vector< std::unique_ptr<Experiment> > experiments;

	// Runtime variables
	VectorReal transformed_variables;
	bcm3::RNG rng;
	VectorReal experiment_likelihood;
};
