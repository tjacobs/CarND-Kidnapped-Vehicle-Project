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

	// Create random distributions
	normal_distribution<double> x_rand(x, std[0]);
	normal_distribution<double> y_rand(y, std[1]);
	normal_distribution<double> theta_rand(theta, std[2]);

	// Create 100 particles
	num_particles = 100;
	for(int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = x_rand(gen);
		p.y = y_rand(gen);
		p.theta = theta_rand(gen);
		p.weight = 1.0;
		particles.push_back(p);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	// Create random distributions
	normal_distribution<double> x_rand(0, std_pos[0]);
	normal_distribution<double> y_rand(0, std_pos[1]);
	normal_distribution<double> theta_rand(0, std_pos[2]);

	// Update particles according to motion model
	for(int i = 0; i < num_particles; i++) {
		if (fabs(yaw_rate) < 0.0001) {
			// Heading straight exception
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}
		else {
			// Add predicted movement to each particle
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}

		// Add the noise
		particles[i].x += x_rand(gen);
		particles[i].y += y_rand(gen);
		particles[i].theta += theta_rand(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the 
	// observed measurement to this particular landmark.

	// For each observation
	for (int i = 0; i < observations.size(); i++) {

		// Get this observation
		LandmarkObs o = observations[i];

		// Reset
		double min_distance = INFINITY;
		int predicted_id = -1;

		// Go through predictions, find closest to this observation
		for (int j = 0; j < predicted.size(); j++) {

			// Get this prediction
			LandmarkObs p = predicted[j];

			// Calculate distance between current and predicted landmarks
			double distance = dist(o.x, o.y, p.x, p.y);

			// Find closest predicted
			if (distance < min_distance) {
				min_distance = distance;
				predicted_id = p.id;
			}
		}

		// Assign this prediction to the observation
		observations[i].id = predicted_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
	 std::vector<LandmarkObs> observations, Map map_landmarks) {

	// Update the weights of each particle using a multi-variate Gaussian distribution.
	for(int i = 0; i < num_particles; i++) {

		// Get particle
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

		// Go through each landmark, find which observations are in range of landmarks
		vector<LandmarkObs> predictions;
		for(int j = 0; j < map_landmarks.landmark_list.size(); j++) {

			// Get landmark
			float landmark_x = map_landmarks.landmark_list[j].x_f;
			float landmark_y = map_landmarks.landmark_list[j].y_f;
			int   landmark_id = map_landmarks.landmark_list[j].id_i;

			// In sensor range? Assume square sensor range
			if(fabs(landmark_x - p_x) <= sensor_range && fabs(landmark_y - p_y) <= sensor_range) {

				// Add prediction
				predictions.push_back(LandmarkObs{ landmark_id, landmark_x, landmark_y });
			}
		}

		// Transform observations into map space
		vector<LandmarkObs> transformed_observations;
		for(int j = 0; j < observations.size(); j++) {
			// From: http://planning.cs.uiuc.edu/node99.html
			// Must flip the X and cos multiply around, for numeric stability
//			double transformed_x = observations[j].x * cos(p_theta) - observations[j].y * sin(p_theta) + p_x;
//			double transformed_y = observations[j].x * sin(p_theta) + observations[j].y * sin(p_theta) + p_y;
			double transformed_x = cos(p_theta) * observations[j].x - sin(p_theta) * observations[j].y + p_x;
			double transformed_y = sin(p_theta) * observations[j].x + cos(p_theta) * observations[j].y + p_y;

			// Add observation
			transformed_observations.push_back(LandmarkObs{ observations[j].id, transformed_x, transformed_y });
		}

		// Associate each particle to map transformed sensor observations
		dataAssociation(predictions, transformed_observations);

		// Reset particle weight
	    particles[i].weight = 1.0;

	    // Go through the observations
		for(int j = 0; j < transformed_observations.size(); j++) {
	      
	        // Get observation
			double o_x = transformed_observations[j].x;
			double o_y = transformed_observations[j].y;
			double pr_x, pr_y;

			// Go through predictions, find the prediction with this observation id
			for(int k = 0; k < predictions.size(); k++) {
				if(predictions[k].id == transformed_observations[j].id) {
					pr_x = predictions[k].x;
					pr_y = predictions[k].y;
					break;
				}
			}

			// Update weight with this horrendus formula
			double s_x = std_landmark[0];
			double s_y = std_landmark[1];
			particles[i].weight *= exp( - (pow(pr_x - o_x, 2) / (2 * pow(s_x, 2) )
									    + (pow(pr_y - o_y, 2) / (2 * pow(s_y, 2) ) ) ) )
								  		/ (2 * M_PI * s_x * s_y);
		}
	}
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 

	// Put all the weights into a vector
	vector<double> weights;
	for (int i = 0; i < num_particles; i++) {
		weights.push_back(particles[i].weight);
	}

	// Go around the wheel (vector) of weights and resample our particles
	vector<Particle> new_particles;
	vector<double> new_weights;
	random_device rd;
	mt19937 gen(rd());
	discrete_distribution<> d(weights.begin(), weights.end());
	for (int i = 0; i<num_particles; i++) {

		// Pick an index, make a new particle with same x, y, theta, weight 1.
	 	int index = d(gen);
		Particle p;
		p.id = i;
		p.x = particles[index].x;
		p.y = particles[index].y;
		p.theta = particles[index].theta;
		p.weight = 1.0;
		new_particles.push_back(p);
	}
	
	// Assign
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
