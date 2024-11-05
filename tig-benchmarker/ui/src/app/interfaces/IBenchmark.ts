export interface IBenchmark {
    id:string;
    age: number;
    challenge_id: string;
    algorithm_id: string;
    solutions: number;
    difficulty: string;
    submission_delay: number;
    qualifiers: number;
    start_time:string;
    end_time:string;
    number_of_nonces:number;
    status:string;
    time_elapsed?:number;
}