import { CameraCapturedPicture } from "expo-camera";
import React, { createContext, useState } from "react";

// Define the initial state of the results object
export interface Result {
	label: string;
	confidence: number;
	image: CameraCapturedPicture | undefined;
	description: string;
}

interface ResultsContextType {
	result?: Result;
	setResult?: React.Dispatch<React.SetStateAction<Result | undefined>>;
}

// Create the ResultsContext
export const ResultsContext = createContext<ResultsContextType | null>(null);

// Create the ResultsProvider component
export const ResultsProvider = ({ children }: React.PropsWithChildren) => {
	const [result, setResult] = useState<Result>();

	// Define any additional functions to update the results object if needed

	return (
		<ResultsContext.Provider value={{ result, setResult }}>
			{children}
		</ResultsContext.Provider>
	);
};

export const useResults = () => {
	const context = React.useContext(ResultsContext);
	if (!context) {
		throw new Error("useResults must be used within a ResultsProvider");
	}
	return context;
};
