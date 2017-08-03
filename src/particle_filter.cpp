/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 * 		Project: Tom Jacobs
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

// Generator
default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

	// Set the number of particles.
	num_particles = 100;


	// Create a Gaussian distribution for x, y
	normal_distribution<double> dist_x(0, std[0]);
	normal_distribution<double> dist_y(0, std[1]);
	normal_distribution<double> dist_theta(0, std[2]);

	// Initialize all particles to first position (based on estimates of 
	// x, y, theta and their uncertainties from GPS) and all weights to 1. 
	for( int i=0; i < num_particles; i++ ) {

		// Add random Gaussian noise to each particle
		Particle p;
		p.x = x + dist_x(gen);
		p.y = y + dist_y(gen);
		p.theta = theta + dist_theta(gen);
		p.weight = 1;
		p.id = i;

		particles.push_back(p);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	// Bring the noise
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_yaw(0, std_pos[2]);

	for (int i = 0; i < num_particles; i++) {

		// Heading straight exception
		if (fabs(yaw_rate) < 0.0001) {
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}
		else {
			// Add measurements to each particle
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}

		// Add the noise
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_yaw(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the 
	// observed measurement to this particular landmark.

	// For each observation
	for (int i = 0; i < observations.size(); i++) {

		// Look at this observation
		LandmarkObs o = observations[i];

		// Reset
		double min_distance = INFINITY;
		int map_id = -1;

		for (unsigned int j = 0; j < predicted.size(); j++) {
			// This prediction
			LandmarkObs p = predicted[j];

			// Distance between current and predicted landmarks
			double distance = dist(o.x, o.y, p.x, p.y);

			// Find closest predicted
			if (distance < min_distance) {
				min_distance = distance;
				map_id = p.id;
			}
		}
		observations[i].id = map_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
	 std::vector<LandmarkObs> observations, Map map_landmarks) {

	// Update the weights of each particle using a multi-variate Gaussian distribution.
	for (int i = 0; i < num_particles; i++) {

		// Get particle
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

		// Predictions
		vector<LandmarkObs> predictions;

		// Go through each landmark
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {

			// Get landmark
			float landmark_x = map_landmarks.landmark_list[j].x_f;
			float landmark_y = map_landmarks.landmark_list[j].y_f;
			int   landmark_id = map_landmarks.landmark_list[j].id_i;

			// In sensor range?
			if (fabs(landmark_x - p_x) <= sensor_range && fabs(landmark_y - p_y) <= sensor_range) {
				// Add prediction
				predictions.push_back(LandmarkObs{ landmark_id, landmark_x, landmark_y });
			}
	    }

		// Transform observations into map space
		vector<LandmarkObs> transformed_observations;
		for (int j = 0; j < observations.size(); j++) {
			double transformed_x = cos(p_theta) * observations[j].x - sin(p_theta) * observations[j].y + p_x;
			double transformed_y = sin(p_theta) * observations[j].x + cos(p_theta) * observations[j].y + p_y;
			transformed_observations.push_back(LandmarkObs{ observations[j].id, transformed_x, transformed_y });
		}

		// Find each particle
		dataAssociation(predictions, transformed_observations);
	    particles[i].weight = 1.0;
		for (int j = 0; j < transformed_observations.size(); j++) {
	      
	        // Get observed
			double o_x, o_y, pr_x, pr_y;
			o_x = transformed_observations[j].x;
			o_y = transformed_observations[j].y;
			int associated_prediction = transformed_observations[j].id;

			// Find
			for (unsigned int k = 0; k < predictions.size(); k++) {
				if (predictions[k].id == associated_prediction) {
					pr_x = predictions[k].x;
					pr_y = predictions[k].y;
				}
			}

			// Update weight
			double s_x = std_landmark[0];
			double s_y = std_landmark[1];
			double obs_w = ( 1/( 2 * M_PI * s_x * s_y) )
							* exp( -(pow(pr_x - o_x, 2) / (2 * pow(s_x, 2) ) 
								   +(pow(pr_y - o_y, 2) / (2 * pow(s_y, 2) ) ) ) );
	    	particles[i].weight *= obs_w;
		}
  	}
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 
	vector<Particle> new_particles;

	// Get those weights into a vector
	vector<double> weights;
	for (int i = 0; i < num_particles; i++) {
		weights.push_back(particles[i].weight);
	}

	// Make some ints
	uniform_int_distribution<int> uniintdist(0, num_particles-1);
	auto index = uniintdist(gen);

	// Get the max of the weights
	double max_weight = *max_element(weights.begin(), weights.end());

	// Make a distribution from 0 to max
	uniform_real_distribution<double> unirealdist(0.0, max_weight);

	// Go through the particles
	double beta = 0.0;
	for (int i = 0; i < num_particles; i++) {
		beta += unirealdist(gen) * 2.0;
		while (beta > weights[index]) {
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		new_particles.push_back(particles[index]);
	}
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	// particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	// Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations = associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
